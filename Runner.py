import numpy as np
from timeit import default_timer as timer
from pyrosetta import *
from statistics import mean
import pandas as pd

from load_ligand import load_ligand, rdkit_to_mutable_res, moltomolblock, generate_conformers, pose_with_ligand, mol_result

import h5py
import json
import os

class Runner():

    def __init__(self, pdbbind_path, pdb, protocol_paths, output_dir):

        # ---       Data setup      ---
        self.pdb = pdb
        print("Start processing", self.pdb, flush=True)

        path = pdbbind_path
        if path[-1] != '/': path += '/'
        path += self.pdb + '/' + self.pdb

        self.pdb_file = path + '_protein.pdb'
        self.lig_file = path + '_ligand.sdf'

        # ---       hdf5 setup      ---
        self.filepath = os.path.join(output_dir, f"{self.pdb}.h5")
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(self.filepath):
            print(f"[{self.pdb}] Resuming existing file...")
        self.file = h5py.File(self.filepath, 'a')
        self.COMPRESSION = "gzip"
        self.COMP_LEVEL = 4
        self.BATCH_SIZE = 50


        # ---       Protocol loading      ---
        # should be a dictionary linking a name to a protocol
        self.protocol_paths = protocol_paths
        self.protocols = {}
        self.scfx = None
        self.score_weights = {}
        self._load_protocols()

        print("Loaded", len(self.protocols), 'protocols', flush=True)

        # store raw energies total and idelta + total score, idelta score, prepare time, rmsd to crystal and rmsd to input
        self.column_names = ['total_score', 'idelta_score', 'rmsd_to_crystal', 'rmsd_to_input', 'prepare_time']
        for name in ['raw_', 'raw_delta_']:
            for energy in self.score_weights:
                self.column_names.append(name + energy)
        self.FLOAT_COUNT = len(self.column_names)

        # ---       Load ligand      ---
        self.conformers = None
        self.mut_res = None
        self.index_to_vd = None
        self.atmname_to_idx = None
        self._prepare_ligand()

        print("Prepared ligand", flush=True)

        # ---       Prepare poses      ---
        self.poses = {
            'crystal' : None,
            'relax' : None,
            'apo_relax' : None,
        }
        self.pose_str_arrays = {
            'crystal' : None,
            'relax' : None,
            'apo_relax' : None,
        }
        self.poses_grp = None
        self.poses_dset_str = None
        self.poses_dset_float = None
        self.poses_dset_rnames = None
        self._prepare_pose_data_structure()

        for key in self.poses:
            self.poses[key] = self._load_pose(key)

        print('Loaded all poses', flush=True)

        # ---       Prepare rmsd calcs      ---

        self.crystal_ligand_rmsd = rosetta.core.simple_metrics.metrics.RMSDMetric()
        self.input_ligand_rmsd = rosetta.core.simple_metrics.metrics.RMSDMetric()
        res_selector = rosetta.core.select.residue_selector.ResidueIndexSelector(self.poses['crystal'].total_residue())
        self.crystal_ligand_rmsd.set_residue_selector(res_selector)
        self.input_ligand_rmsd.set_residue_selector(res_selector)
        self.crystal_ligand_rmsd.set_comparison_pose(self.poses['crystal'])

        print('Prepared rmsd calculator', flush=True)

        # ---       Prepare protein ligand complex datastructure      ---
        self.docking_dset_str = None
        self.docking_dset_float = None
        self.docking_dset_protocols = None
        self._prepare_complex_data_structure()

        print('Prepared docking data structure', flush=True)



    def run(self, n_relax, n_apo_relax, n_dock):
        total_start = timer()
        self.process_pose('crystal', n_relax=0)
        self.process_pose('relax', n_relax=n_relax)
        self.process_pose('apo_relax', n_relax=n_apo_relax)

        # load minimal score results and save them to self.poses
        df = pd.DataFrame(self.poses_dset_float, columns=self.column_names)
        df.insert(loc=0, column='type', value=self.poses_dset_rnames.asstr()[()])
        for name in ['crystal', 'relax', 'apo_relax']:
            min_idx = df[df.type == name]['total_score'].idxmin()
            pdb_str = self.poses_dset_str.asstr()[min_idx]
            score = df.loc[min_idx]['total_score']
            idelta = df.loc[min_idx]['idelta_score']
            rmsd = df.loc[min_idx]['rmsd_to_crystal']
            conf = mol_result(self.conformers, pdb_str, self.atmname_to_idx)
            self.poses[name] = pose_with_ligand(pdb_str, conf)
            #self.poses[name].dump_pdb(name + '.pdb')
            #with open(name + '2.pdb', 'w') as file:
            #    file.write(pdb_str)
            self.pose_str_arrays[name] = pdb_str
            print(f"{name}: Loaded pose {min_idx} with score {score:.4f} idelta {idelta:.4f} rmsd {rmsd:.4f}")

        for name in ['crystal', 'relax', 'apo_relax']:
            for protocol in self.protocols:
                self.dock(name, protocol, n_dock)

        end = timer()
        time = end - total_start
        print(f"Finished everything in {time/60:.4f} minutes", flush=True)

    def dock(self, name, protocol, n_docking):
        data = self.docking_dset_protocols[()]
        query = name + '_' + protocol
        query = query.encode('utf-8')
        count = np.count_nonzero(data == query)

        col_index = {
            key : np.where(self.column_names == key)[0][0] for key in self.column_names
        }

        print(f"Start docking {name} {protocol}. {count} runs are recorded, {n_docking - count} remain", flush=True)
        
        start = timer()

        compressions = []
        total_scores = []
        idelta_scores = []
        crystal_rmsds = []
        input_rmsds = []

        batch_results = []
        batch_protocols = []
        batch_update_str = []

        buffer_size = 10
        self.input_ligand_rmsd.set_comparison_pose(self.poses[name])
        
        while count < n_docking:
            count += 1
            pose, results = self.single_dock(self.poses[name].clone(), self.protocols[protocol])

            update_strings = {}
            pdb_stringarr = self.pose_to_stringarr(pose)
            orig_stringarr = self.pose_str_arrays[name]
            for i,line in enumerate(pdb_stringarr):
                if line != orig_stringarr[i]:
                    update_strings[i] = line
            compressions.append(1 - (len(update_strings)/len(orig_stringarr)))

            total_scores.append(results[col_index['total_score']])
            idelta_scores.append(results[col_index['idelta_score']])
            crystal_rmsds.append(results[col_index['rmsd_to_crystal']])
            input_rmsds.append(results[col_index['rmsd_to_input']])

            batch_results.append(results)
            batch_protocols.append(name + '_' + protocol)
            batch_update_str.append(json.dumps(update_strings))

            if len(batch_results) >= buffer_size:
                self.store_complex_results(batch_results, batch_protocols, batch_update_str)
                batch_results = []
                batch_protocols = []
                batch_update_str = []

        if batch_results:
            self.store_complex_results(batch_results, batch_protocols, batch_update_str)
            
        end = timer()
        time = end - start

        if total_scores:
            print(f"\tBest score: {min(total_scores):.4f}")
            print(f"\tBest idelta: {min(idelta_scores):.4f}")
            print(f"\tBest rmsd to crystal: {min(crystal_rmsds):.4f}")
            print(f"\tBest rmsd to input: {min(input_rmsds):.4f}")
            print(f"\tAverage pdb size reduction: {100*mean(compressions):.4f}%")
        print(f"\tFinished docking in {time/60:.4f} minutes", flush=True)

    def store_complex_results(self, batch_results, batch_protocols, batch_update_str):
        # Resize data sets
        n_new = len(batch_results)
        current_size = self.docking_dset_float.shape[0]
        new_size = current_size + n_new
        self.docking_dset_float.resize((new_size, self.FLOAT_COUNT))
        self.docking_dset_str.resize((new_size,))
        self.docking_dset_protocols.resize((new_size,))

        # write data
        self.docking_dset_float[current_size : new_size] = batch_results
        self.docking_dset_str[current_size : new_size] = batch_update_str
        self.docking_dset_protocols[current_size : new_size] = batch_protocols

        # flush
        self.file.flush()

    def single_dock(self, pose, protocol):
        results = np.zeros(shape=len(self.column_names), dtype='float32')

        col_index = {
            key : np.where(self.column_names == key)[0][0] for key in self.column_names
        }

        start = timer()

        protocol.apply(pose)
        results[col_index['total_score']] = self.scfx(pose)
        interface_scores = rosetta.protocols.ligand_docking.get_interface_deltas( 'X', pose, self.scfx )
        results[col_index['idelta_score']] = interface_scores["interface_delta_X"]
        results[col_index['rmsd_to_crystal']] = self.crystal_ligand_rmsd.calculate(pose)
        results[col_index['rmsd_to_input']] = self.input_ligand_rmsd.calculate(pose)

        for score_type in interface_scores.keys():
            energy = str(score_type)[5:]
            if energy in self.score_weights:
                term = 'raw_delta_' + energy
                weight = self.score_weights[energy]
                raw_energy = interface_scores[score_type] / weight
                results[col_index[term]] = raw_energy

        for i in range(rosetta.core.scoring.n_score_types):
            weight = self.scfx.weights()[ rosetta.core.scoring.ScoreType(i) ]
            term = 'raw_' + str(rosetta.core.scoring.ScoreType(i)).split('.')[-1]
            raw_energy = pose.energies().total_energies()[ rosetta.core.scoring.ScoreType(i) ]
            if abs(weight) > 1e-100:
                results[col_index[term]] = raw_energy

        end = timer()
        results[col_index['prepare_time']] = end - start

        return pose, results

    def process_pose(self, name, n_relax):
        data = self.poses_dset_rnames[()]
        query = name.encode('utf-8')
        count = np.count_nonzero(data == query)
        remaining_relax = n_relax - count
        if remaining_relax > 0:
            print(f"Found {count} relaxed versions of {name}. Start to relax {remaining_relax} more.", flush=True)
            while count < n_relax:
                count += 1
                print(f"Relax {count}/{n_relax}")
                pose, results = self.calculate_pose(self.poses['crystal'].clone(), name, relax=True)
                self.store_pose_results(pose, results, name)
                print("\tResults are saved to file.")
        elif count == 0:
            print(f"Processing {name} without relax.")
            pose, results = self.calculate_pose(self.poses['crystal'].clone(), name, relax=False)
            self.store_pose_results(pose, results, name)
            print("\tResults are saved to file.")

        print(f"Finished processing {name}", flush=True)

    def close(self):
        if self.file:
            self.file.close()

    def store_pose_results(self, pose, results, name):
        # Resize data sets
        new_size = self.poses_dset_float.shape[0] + 1
        self.poses_dset_float.resize((new_size, self.FLOAT_COUNT))
        self.poses_dset_rnames.resize((new_size,))
        self.poses_dset_str.resize((new_size,))

        # write data
        self.poses_dset_float[new_size - 1] = results
        self.poses_dset_rnames[new_size - 1] = name
        self.poses_dset_str[new_size - 1] = "\n".join(self.pose_to_stringarr(pose))

        # flush
        self.file.flush()

    def pose_to_stringarr(self, pose):
        pu = rosetta.core.io.pose_to_sfr.PoseToStructFileRepConverter()
        pu.init_from_pose(pose)
        string = rosetta.core.io.pdb.create_pdb_contents_from_sfr(pu.sfr())
        stringarr = []
        for line in string.split('\n'):
            if len(line) == 0 : continue
            if line[0] == '#': break
            #if line[:4] == 'ATOM' or line[:3] == 'TER' or line[:6] == 'HETATM':
            stringarr.append(line)
        return stringarr
    
    def calculate_pose(self, pose, name, relax):

        results = np.zeros(shape=len(self.column_names), dtype='float32')

        col_index = {
            key : np.where(self.column_names == key)[0][0] for key in self.column_names
        }

        start = timer()
        
        if name == 'apo_relax':
            res_array = pyrosetta.rosetta.utility.vector1_bool(pose.total_residue())
            res_array[pose.total_residue()] = True
            jump_id = pyrosetta.rosetta.core.kinematics.jump_which_partitions( pose.fold_tree(), res_array )
            if jump_id == 0:
                raise ValueError("Faulty jump id")
            mover = pyrosetta.rosetta.protocols.rigid.RigidBodyTransMover( pose, jump_id )
            trans_vec = mover.trans_axis()
            mover.step_size(500)
            mover.apply(pose)

        if relax:
            fast_relax = rosetta.protocols.relax.FastRelax()
            fast_relax.set_scorefxn(self.scfx)
            fast_relax.ramp_down_constraints(False)

            coord_cst_mover = rosetta.protocols.relax.AtomCoordinateCstMover()
            coord_cst_mover.cst_sidechain( False )
            coord_cst_mover.ambiguous_hnq( True )
            coord_cst_mover.apply(pose)

            fast_relax.apply(pose)

        # saving total score
        results[col_index['total_score']] = self.scfx(pose)
        print(f"\tSaved total score: {results[col_index['total_score']]:.4f}")

        # saving interface scores
        interface_scores = rosetta.protocols.ligand_docking.get_interface_deltas( 'X', pose, self.scfx )
        results[col_index['idelta_score']] = interface_scores["interface_delta_X"]
        print(f"\tSaved interface delta score: {results[col_index['idelta_score']]:.4f}")
        counter = 0
        for score_type in interface_scores.keys():
            energy = str(score_type)[5:]
            if energy in self.score_weights:
                term = 'raw_delta_' + energy
                weight = self.score_weights[energy]
                raw_energy = interface_scores[score_type] / weight
                results[col_index[term]] = raw_energy
                counter += 1
        print(f"\tSaved {counter} raw delta Rosetta energy terms")

        # saving raw energies
        counter = 0
        for i in range(rosetta.core.scoring.n_score_types):
            weight = self.scfx.weights()[ rosetta.core.scoring.ScoreType(i) ]
            term = 'raw_' + str(rosetta.core.scoring.ScoreType(i)).split('.')[-1]
            raw_energy = pose.energies().total_energies()[ rosetta.core.scoring.ScoreType(i) ]
            if abs(weight) > 1e-100:
                results[col_index[term]] = raw_energy
                counter += 1
        print(f"\tSaved {counter} raw Rosetta energy terms")

        if name == 'apo_relax':
            rmsd_prior = self.crystal_ligand_rmsd.calculate(pose)
            print("\tRMSD prior replacement at binding site:", f'{rmsd_prior:.4f}')
            mover.trans_axis(trans_vec.negate())
            mover.apply(pose)
            lig_replaced_score = self.scfx(pose)
            print("\tMoved ligand back into binding pocket. New score:", f'{lig_replaced_score:.4f}')

        rmsd = self.crystal_ligand_rmsd.calculate(pose)
        results[col_index['rmsd_to_crystal']] = rmsd
        print("\tRMSD to crystall structure:", f'{rmsd:.4f}')

        had_constraints = pose.remove_constraints()
        if had_constraints:
            print("\tRemoved all constraints from pose")
        else:
            print("\tNo constraints were added to pose.")

        end = timer()
        results[col_index['prepare_time']] = end - start
        print(f"\tProcessing this pose took {results[col_index['prepare_time']]/60:.4f} minutes", flush=True)

        return pose, results

    def _load_pose(self, name):
        if name != 'crystal' and self.poses['crystal'] != None:
            return self.poses['crystal'].clone()
        
        pdb_string = ''
        with open(self.pdb_file) as file:
            pdb_string = file.read()

        pose = pose_with_ligand(pdb_string, self.conformers, mut_res=self.mut_res, index_to_vd=self.index_to_vd)

        return pose

    def _prepare_ligand(self):
        mol = load_ligand(self.lig_file)
        self.conformers = generate_conformers(mol, seed=12345)
        self.mut_res, self.index_to_vd = rdkit_to_mutable_res(self.conformers)

        if 'ligand_sdf' not in self.file:
            dset_sdf = self.file.create_dataset(
                'ligand_sdf', 
                shape=(), 
                dtype=h5py.string_dtype(encoding='utf-8'),
            )
        else:
            dset_sdf = self.file['ligand_sdf']
        dset_sdf[()] = moltomolblock(self.conformers)

        self.atmname_to_idx = {}
        for idx, vd in self.index_to_vd.items():
            name = self.mut_res.atom_name(vd)
            self.atmname_to_idx[ name ] = idx
        if 'atmname_to_idx' not in self.file:
            dset_map = self.file.create_dataset(
                'atmname_to_idx',
                shape=(),
                dtype=h5py.string_dtype(encoding='utf-8'),
            )
        else:
            dset_map = self.file['atmname_to_idx']
        dset_map[()] = json.dumps(self.atmname_to_idx)

    def _prepare_complex_data_structure(self):
        if 'pdb_strings' not in self.file:
            self.docking_dset_str = self.file.create_dataset(
                "pdb_strings", 
                shape=(0,), 
                maxshape=(None,), 
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(self.BATCH_SIZE,),
                compression=self.COMPRESSION, 
                compression_opts=self.COMP_LEVEL
            )
        else:
            self.docking_dset_str = self.file['pdb_strings']
        if 'results' not in self.file: 
            self.docking_dset_float = self.file.create_dataset(
                "results", 
                shape=(0, self.FLOAT_COUNT), 
                maxshape=(None, self.FLOAT_COUNT), 
                dtype='float32',
                chunks=(self.BATCH_SIZE, self.FLOAT_COUNT),
                compression=self.COMPRESSION, 
                compression_opts=self.COMP_LEVEL,
                shuffle=True
            )
            self.docking_dset_float.attrs['column_names'] = self.column_names
        else:
            self.docking_dset_float = self.file['results']
        if 'protocol' not in self.file:
            self.docking_dset_protocols = self.file.create_dataset(
                "protocol", 
                shape=(0,), 
                maxshape=(None,), 
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(self.BATCH_SIZE,),
                compression=self.COMPRESSION, 
                compression_opts=self.COMP_LEVEL
            )
        else:
            self.docking_dset_protocols = self.file['protocol']

    def _prepare_pose_data_structure(self):
        if 'poses' not in self.file:
            self.poses_grp = self.file.create_group("poses")
            self.poses_dset_str = self.poses_grp.create_dataset(
                "pdb_strings", 
                shape=(0,), 
                maxshape=(None,), 
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(1,),
                compression=self.COMPRESSION, 
                compression_opts=self.COMP_LEVEL
            )
            self.poses_dset_float = self.poses_grp.create_dataset(
                "results", 
                shape=(0, self.FLOAT_COUNT), 
                maxshape=(None, self.FLOAT_COUNT), 
                dtype='float32',
                chunks=(1, self.FLOAT_COUNT),
                compression=self.COMPRESSION, 
                compression_opts=self.COMP_LEVEL,
                shuffle=True
            )
            self.poses_dset_float.attrs['column_names'] = self.column_names
            self.poses_dset_rnames = self.poses_grp.create_dataset(
                "row_names", 
                shape=(0,), 
                maxshape=(None,), 
                dtype=h5py.string_dtype(encoding='utf-8'),
                chunks=(1,),
                compression=self.COMPRESSION, 
                compression_opts=self.COMP_LEVEL
            )
        else:
            self.poses_grp = self.file['poses']
            self.poses_dset_str = self.file['poses']['pdb_strings']
            self.poses_dset_float = self.file['poses']['results']
            self.poses_dset_rnames = self.file['poses']['row_names']
        
        self.column_names = self.poses_dset_float.attrs['column_names']
        
    def _load_protocols(self):
        self.protocols = {}
        for path in self.protocol_paths:
            name = path.split('/')[-1].split('.')[0]
            print("Loading protocol", path, 'as', name)
            xml_objects = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file(path)
            self.protocols[name] = ( xml_objects.get_mover("ParsedProtocol") )
            if self.scfx == None:
                print("[Warning] Trying to load a score function called \"hard_rep\" from this xml protocol. It will overwrite the previously loaded score function and will be used to score everything outside of protocols. This behaviour might be undesired.")
                self.scfx = xml_objects.get_score_function("hard_rep")
                self.scfx.set_weight( rosetta.core.scoring.coordinate_constraint, 0.5 )

        for i in range(rosetta.core.scoring.n_score_types):
            weight = self.scfx.weights()[ rosetta.core.scoring.ScoreType(i) ]
            term = str(rosetta.core.scoring.ScoreType(i)).split('.')[-1]
            if abs(weight) > 1e-100:
                self.score_weights[ term ] = weight
        if 'scfx_weights' not in self.file:
            dset_weights = self.file.create_dataset(
                'scfx_weights', 
                shape=(), 
                dtype=h5py.string_dtype(encoding='utf-8'),
            )
        else:
            dset_weights = self.file['scfx_weights']
        dset_weights[()] = json.dumps(self.score_weights)


if __name__ == '__main__':
    
    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:100', silent=True)

    try:
        r = Runner(
            pdbbind_path='pdbbind_cleaned',
            pdb='5k00',
            protocol_paths=['xml_protocols/docking_std.xml'],
            output_dir='h5_test'
        )
        r.run(
            n_relax=2,
            n_apo_relax=0,
            n_dock=7
        )
    finally:
        r.close()