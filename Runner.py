import numpy as np
from timeit import default_timer as timer
from pyrosetta import *
from statistics import mean

from load_ligand import load_ligand, rdkit_to_mutable_res, moltomolblock, generate_conformers, molfrommolblock, pose_with_ligand

from rdkit.Geometry import Point3D

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
        self._prepare_ligand()

        print("Prepared ligand", flush=True)

        # ---       Prepare poses      ---
        self.poses = {
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



    def run(self, n_relax, n_apo_relax, n_dock):
        total_start = timer()

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
            dset_sdf = self.file.create_dataset('ligand_sdf', shape=(), dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            dset_sdf = self.file['ligand_sdf']
        dset_sdf[()] = moltomolblock(self.conformers)

        atmname_to_idx = {}
        for idx, vd in self.index_to_vd.items():
            name = self.mut_res.atom_name(vd)
            atmname_to_idx[ name ] = idx
        if 'atmname_to_idx' not in self.file:
            dset_map = self.file.create_dataset('atmname_to_idx', shape=(), dtype=h5py.string_dtype(encoding='utf-8'),)
        else:
            dset_map = self.file['atmname_to_idx']
        dset_map[()] = json.dumps(atmname_to_idx)

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
            self.column_names = self.poses_dset_float.attrs['column_names']
            self.poses_dset_rnames = self.file['poses']['row_names']
        
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
            dset_weights = self.file.create_dataset('scfx_weights', shape=(), dtype=h5py.string_dtype(encoding='utf-8'))
        else:
            dset_weights = self.file['scfx_weights']
        dset_weights[()] = json.dumps(self.score_weights)


if __name__ == '__main__':
    
    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:100', silent=True)

    r = Runner(
        pdbbind_path='pdbbind_cleaned',
        pdb='1a0q',
        protocol_paths=['xml_protocols/docking_std.xml'],
        output_dir='h5_test'
    )