import numpy as np
from timeit import default_timer as timer
from pyrosetta import *
from statistics import mean

from load_ligand import load_ligand, rdkit_to_mutable_res, add_confs_to_res, mutable_res_to_res

class Runner():

    def __init__(self, pdbbind_path, pdb, protocol_paths):

        # ---       Data setup      ---
        self.pdb = pdb

        path = pdbbind_path
        if path[-1] != '/': path += '/'
        path += self.pdb + '/' + self.pdb

        self.pdb_file = path + '_protein.pdb'
        self.lig_file = path + '_ligand.mol2'

        # ---       Protocol loading      ---
        # should be a dictionary linking a name to a protocol
        self.protocol_paths = protocol_paths
        self.protocols = {}
        self.scfx = None
        self.score_weights = {}

        # ---       Pose and conformer loading      ---
        self.poses = {
            # unchanged pose from pdb file
            'pose' : rosetta.core.pose.Pose(),
            # relaxed pose with all cofactors
            'pose_relax' : None,
            # unchanged pose from pdb file, but ligand got move before scoring - should not be docked into, since it is the same as pose
            'pose_ligaway' : None,
            # relaxed pose with ligand moved outside of pocket before relax
            'pose_relax_ligaway' : None,
        }

        self.crystal_ligand_rmsd = rosetta.core.simple_metrics.metrics.RMSDMetric()
        self.input_ligand_rmsd = rosetta.core.simple_metrics.metrics.RMSDMetric()
        self.best_score_ligand_rmsd = rosetta.core.simple_metrics.metrics.RMSDMetric()
        self.best_idelta_ligand_rmsd = rosetta.core.simple_metrics.metrics.RMSDMetric()

        # Conformers are stored in case someone wants to analyse available conformers during docking
        self.conformers = None
        # Maps from atom name used by Rosetta to RDKit mol index. This should help to recover the rdkit mol with docked coordinates. One of the mols saved in self.conformers should be used
        self.atmname_to_idx = None

        # ---       Store pose results      ---
        # The ligaway pdb records differ from the poses. The ligands remain moved away from protein in the pdb, but are returned to their original location in the pose
        # this is important for docking, since the initial ligand position is the start position
        self.complex_results = {
            'pose' : None,
            'pose_relax' : None,
            'pose_ligaway' : None,
            'pose_relax_ligaway' : None,
        }
        self.n_relax = None

        # ---       Store run results      ---
        # should hold an array of results for each protocol name and each starting structure
        # result arrays should hold 150 entries, each entry consisting of
        #       - pose string delta
        #       - total score
        #       - raw energy terms
        #       - rmsd to crystal, crystal relax, lowest energy run results, lowest i delta run results
        #       - runtime
        self.docking_results = {}

    def run(self, n_relax, n_runs):
        total_start = timer()
        self.n_relax = n_relax
        self.load_protocols()
        self.load_poses()
        self.store_poses()

        if n_runs > 0:
            for protocol_name, protocol in self.protocols.items():
                for pose_name, pose in self.poses.items():
                    if pose_name == "pose_ligaway":
                        # this pose will be skipped since it is the same as the crystal structure
                        # it is only generated to calculate delta scores of the crystal structure
                        continue
                    
                    # run protocols
                    run_start = timer()
                    run_name = protocol_name + "_" + pose_name
                    print("Start", n_runs, "repeats of", run_name)
                    result_poses = []
                    result_times = []
                    best_score = 99999.9
                    best_score_pose = None
                    best_idelta = 99999.9
                    best_idelta_pose = None
                    for i in range(n_runs):
                        start = timer()
                        work_pose = pose.clone()
                        protocol.apply(work_pose)
                        end = timer()
                        result_times.append(end - start)
                        result_poses.append(work_pose)
                        score = self.scfx(work_pose)
                        idelta = rosetta.protocols.ligand_docking.get_interface_deltas( 'X', work_pose, self.scfx )["interface_delta_X"]
                        if score < best_score:
                            best_score = score
                            best_score_pose = work_pose
                        if idelta < best_idelta:
                            best_idelta = idelta
                            best_idelta_pose = work_pose

                    # process results
                    self.input_ligand_rmsd.set_comparison_pose(pose)
                    self.best_score_ligand_rmsd.set_comparison_pose(best_score_pose)
                    self.best_idelta_ligand_rmsd.set_comparison_pose(best_idelta_pose)
                    self.docking_results[run_name] = []
                    compressions = []
                    for i in range(len(result_poses)):
                        result, compression = self.store_docking_result(pose_name, result_poses[i], result_times[i])
                        self.docking_results[run_name].append(result)
                        compressions.append(compression)

                    run_end = timer()
                    print("\tFinished docking in", f'{(run_end - run_start)/60:.4f}', "minutes")
                    print("\tBest score:", f'{best_score:.4f}', "RMSD to input:", f'{self.input_ligand_rmsd.calculate(best_score_pose):.4f}', "RMSD to crystal:", f'{self.crystal_ligand_rmsd.calculate(best_score_pose):.4f}')
                    print("\tBest idelta:", f'{best_idelta:.4f}', "RMSD to input:", f'{self.input_ligand_rmsd.calculate(best_idelta_pose):.4f}', "RMSD to crystal:", f'{self.crystal_ligand_rmsd.calculate(best_idelta_pose):.4f}')
                    print("\tAverage pdb size reduction:", f'{100*mean(compressions):.4f}%')

                    best_idelta_pose.dump_pdb(run_name + "_idelta.pdb")
                    best_score_pose.dump_pdb(run_name + "_score.pdb")
        else:
            print("[WARNING] No docking runs are conducted. Set n_runs > 0 to change that.")

        total_end = timer()
        print("Finished processing of", self.pdb, "in", f'{(total_end - total_start)/60:.4f}', "minutes")

    def store_docking_result(self, input_pose_name, result_pose, time):
        #       - pose string delta
        #       - total score, idelta
        #       - raw energy terms
        #       - rmsd to crystal, crystal relax, lowest energy run results, lowest i delta run results
        #       - runtime
        results = {
            "input_pose_name" : input_pose_name,
            "pdb_string_delta" : {},
            "total_score" : None,
            "idelta_score" : None,
            "raw_energies" : {},
            "raw_delta_energies" : {},
            "rmsd_to_crystal" : None,
            "rmsd_to_input" : None,
            "rmsd_to_lowest_score" : None,
            "rmsd_to_lowest_idelta" : None,
        }

        # saving pdb string deltas
        pdb_stringarr = self.pose_to_stringarr(result_pose)
        orig_stringarr = self.complex_results[input_pose_name]['pdb_string_arr']
        for i,line in enumerate(pdb_stringarr):
            if line != orig_stringarr[i]:
                results['pdb_string_delta'][i] = line
        compression = 1 - (len(results['pdb_string_delta'])/len(orig_stringarr))

        # saving scores
        results['total_score'] = self.scfx(result_pose)
        interface_scores = rosetta.protocols.ligand_docking.get_interface_deltas( 'X', result_pose, self.scfx )
        results['idelta_score'] = interface_scores["interface_delta_X"]
        for score_type in interface_scores.keys():
            term = str(score_type)[5:]
            if term in self.score_weights:
                weight = self.score_weights[term]
                results['raw_delta_energies'][term] = interface_scores[score_type] / weight

        # saving raw energies
        for i in range(rosetta.core.scoring.n_score_types):
            weight = self.scfx.weights()[ rosetta.core.scoring.ScoreType(i) ]
            term = str(rosetta.core.scoring.ScoreType(i)).split('.')[-1]
            raw_energy = result_pose.energies().total_energies()[ rosetta.core.scoring.ScoreType(i) ]
            if abs(weight) > 1e-100:
                results[ 'raw_energies' ][ term ] = raw_energy

        # saving rmsds
        results['rmsd_to_crystal'] = self.crystal_ligand_rmsd.calculate(result_pose)
        results['rmsd_to_input'] = self.input_ligand_rmsd.calculate(result_pose)
        results['rmsd_to_lowest_score'] = self.best_score_ligand_rmsd.calculate(result_pose)
        results['rmsd_to_lowest_idelta'] = self.best_idelta_ligand_rmsd.calculate(result_pose)

        return results, compression


    def load_protocols(self):
        self.protocols = {}
        for path in self.protocol_paths:
            name = path.split('/')[-1].split('.')[0]
            print("Loading protocol", path, 'as', name)
            xml_objects = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file(path)
            self.protocols[name] = ( xml_objects.get_mover("ParsedProtocol") )
            print("[Warning] Trying to load a score function called \"hard_rep\" from this xml protocol. It will overwrite the previously loaded score function and will be used to score everything outside of protocols. This behaviour might be undesired.")
            self.scfx = xml_objects.get_score_function("hard_rep")
        
        for i in range(rosetta.core.scoring.n_score_types):
            weight = self.scfx.weights()[ rosetta.core.scoring.ScoreType(i) ]
            term = str(rosetta.core.scoring.ScoreType(i)).split('.')[-1]
            if abs(weight) > 1e-100:
                self.score_weights[ term ] = weight
        
    def load_poses(self):
        pdb_string = ""
        with open(self.pdb_file) as file:
            pdb_string = file.read()

        rosetta.core.import_pose.pose_from_pdbstring(self.poses['pose'], pdb_string)

        self.conformers = load_ligand(self.lig_file)
        mut_res, index_to_vd = rdkit_to_mutable_res(self.conformers)
        self.generate_rosetta_atmname_to_index(index_to_vd, mut_res)
        mut_res = add_confs_to_res(self.conformers, mut_res, index_to_vd)
        res = mutable_res_to_res(mut_res)

        self.poses['pose'].append_residue_by_jump( res, 1, "", "", True )
        self.poses['pose'].pdb_info().chain( self.poses['pose'].total_residue(), 'X' )
        self.poses['pose'].update_pose_chains_from_pdb_chains()

        self.poses['pose_relax'] = self.poses['pose'].clone()
        self.poses['pose_ligaway'] = self.poses['pose'].clone()
        self.poses['pose_relax_ligaway'] = self.poses['pose'].clone()

        res_selector = rosetta.core.select.residue_selector.ResidueIndexSelector(self.poses['pose'].total_residue())
        self.crystal_ligand_rmsd.set_residue_selector(res_selector)
        self.input_ligand_rmsd.set_residue_selector(res_selector)
        self.best_score_ligand_rmsd.set_residue_selector(res_selector)
        self.best_idelta_ligand_rmsd.set_residue_selector(res_selector)

        self.crystal_ligand_rmsd.set_comparison_pose(self.poses['pose'])

        # generate original pdb string for reference
        self.orig_pdb_string = self.pose_to_stringarr(self.poses['pose'])

    def generate_rosetta_atmname_to_index(self, index_to_vd, mut_res):
        atmname_to_idx = {}
        for idx, vd in index_to_vd.items():
            name = mut_res.atom_name(vd)
            atmname_to_idx[ name ] = idx
        self.atmname_to_idx = atmname_to_idx

    def store_poses(self):
        for name, pose in self.poses.items():
            print("Start processing pose:", name)
            relax = 'relax' in name
            move_ligand = 'ligaway' in name
            self.complex_results[name], self.poses[name] = self.process_pose(pose, relax=relax, move_ligand=move_ligand)
            self.poses[name].dump_pdb(name + ".pdb")

    def process_pose(self, in_pose, relax, move_ligand):
        results = {
            'pdb_string_arr' : [],
            'total_score' : None,
            'raw_energies' : {},
            'rmsd_to_crystal' : None,
            'prepare_time' : None,
            # will only be filled if relax is called
            'score_distribution' : None,
        }

        start = timer()
        
        if move_ligand:
            res_array = pyrosetta.rosetta.utility.vector1_bool(in_pose.total_residue())
            res_array[in_pose.total_residue()] = True
            jump_id = pyrosetta.rosetta.core.kinematics.jump_which_partitions( in_pose.fold_tree(), res_array )
            if jump_id == 0:
                raise ValueError("Faulty jump id")
            mover = pyrosetta.rosetta.protocols.rigid.RigidBodyTransMover( in_pose, jump_id )
            trans_vec = mover.trans_axis()
            mover.step_size(500)
            mover.apply(in_pose)

        if relax:
            fast_relax = rosetta.protocols.relax.FastRelax()
            fast_relax.set_scorefxn(self.scfx)
            fast_relax.constrain_relax_to_start_coords(True)
            fast_relax.coord_constrain_sidechains(not move_ligand)
            fast_relax.ramp_down_constraints(False)
            scores = []
            max_relax = self.n_relax
            best_pose = in_pose
            for i in range(max_relax):
                work_pose = in_pose.clone()
                fast_relax.apply(work_pose)
                score = self.scfx(work_pose)
                rmsd = self.crystal_ligand_rmsd.calculate(work_pose)
                scores.append(score)
                if score <= min(scores):
                    best_pose = work_pose
                print("\t\tFinished relax", i+1, "of", max_relax, "- Score:", f'{score:.4f}', "RMSD:", f'{rmsd:.4f}')

            in_pose = best_pose
            results['score_distribution'] = scores

        results['pdb_string_arr'] = self.pose_to_stringarr(in_pose)

        # saving total score
        results['total_score'] = self.scfx(in_pose)
        print("\tSaved total score:", f'{results["total_score"]:.4f}')

        # saving raw energies
        for i in range(rosetta.core.scoring.n_score_types):
            weight = self.scfx.weights()[ rosetta.core.scoring.ScoreType(i) ]
            term = str(rosetta.core.scoring.ScoreType(i)).split('.')[-1]
            raw_energy = in_pose.energies().total_energies()[ rosetta.core.scoring.ScoreType(i) ]
            if abs(weight) > 1e-100:
                results[ 'raw_energies' ][ term ] = raw_energy
        print("\tSaved", len(results['raw_energies']), 'raw Rosetta energy terms')

        if move_ligand:
            rmsd_prior = self.crystal_ligand_rmsd.calculate(in_pose)
            print("\tRMSD prior replacement at binding site:", f'{rmsd_prior:.4f}')
            mover.trans_axis(trans_vec.negate())
            mover.apply(in_pose)
            lig_replaced_score = self.scfx(in_pose)
            print("\tMoved ligand back into binding pocket. New score:", f'{lig_replaced_score:.4f}')

        rmsd = self.crystal_ligand_rmsd.calculate(in_pose)
        results['rmsd_to_crystal'] = rmsd
        print("\tRMSD to crystall structure:", f'{rmsd:.4f}')

        end = timer()
        results['prepare_time'] = end - start
        print("\tProcessing this pose took", f'{results["prepare_time"]:.4f}', 'seconds')

        return results, in_pose

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