# This is a helper script to easily access docking results

import zarr
from rdkit import Chem
from rdkit.Geometry import Point3D
import copy

class PDBBindDock:

    def __init__(self, zarr_path):
        
        self.results = zarr.open(zarr_path, mode='r')

        self.pdbs = list(self.results.keys())
        self.docking_protocols = list(self.results[self.pdbs[0]]['docking_results'].keys())
        

    def pdb_result(self, pdb, protocol, run_index):
        res = self.results[pdb]['docking_results'][protocol][run_index]
        base_pdb_strarr = self.results[pdb]['complex_results'][res['input_pose_name']][0]['pdb_string_arr']
        for idx, upd in res['pdb_string_delta'].items():
            base_pdb_strarr[int(idx)] = upd
        return "\n".join(base_pdb_strarr)
    
    def sanitized_mol(self, pdb):
        molblock = self.results[pdb]['sanitized_sdf']
        molblock = "\n".join(molblock[:])
        return Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)

    def mol_result(self, pdb, protocol, run_index):
        pdb_str = self.pdb_result(pdb, protocol, run_index)

        sanitized_mol = self.sanitized_mol(pdb)

        for line in pdb_str:
            if line[:6] == 'HETATM' and line[17:20] == 'UNK':
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                name = line[12:16]
                sanitized_mol.GetConformer().SetAtomPosition(self.results[pdb]['atmname_to_idx'][0][name], Point3D(x,y,z))

        return sanitized_mol
        
    def write_mol(self, filename, mol):
        with Chem.SDWriter(filename) as f:
            f.SetKekulize(False)
            f.write(mol)

    def write_pdb(self, filename, pdb_str):
        with open(filename, 'w') as file:
            file.write(pdb_str)

    def docking_results_statistics(self, pdb, protocol):
        protocol_docking_results = self.results[pdb]['docking_results'][protocol]
        indices = list(range(0, len(protocol_docking_results)))
        total_scores = [res['total_score'] for res in protocol_docking_results]
        idelta_scores = [res['idelta_score'] for res in protocol_docking_results]
        rmsds_to_crystal = [res['rmsd_to_crystal'] for res in protocol_docking_results]
        rmsds_to_input = [res['rmsd_to_input'] for res in protocol_docking_results]

        return indices, total_scores, idelta_scores, rmsds_to_crystal, rmsds_to_input