from pathlib import Path
import h5py
import json
import pandas as pd
from rdkit import Chem
import copy

from load_ligand import mol_result

def pdb_result( orig_pdb_str, update_pdb_strarr ):
    """
    This function generates a full pdb string from original and update.
    The original pdb is an array where every entry is a line from the original pdb file.
    The update is a dictionary with line numbers as keys and it holds replacement to the original pdb for each line that changed during docking.
    """
    orig_pdb_strarr = orig_pdb_str.split('\n')
    result = copy.deepcopy( orig_pdb_strarr )
    for idx, upd in update_pdb_strarr.items():
        result[int(idx)] = upd
    return "\n".join(result)


class Result:

    def __init__(self, h5_file_path: str, name: str) -> None:
        self.name: str = name
        self.path: str = h5_file_path

        with h5py.File(self.path, 'r') as file:
            self.atmname_to_index: dict[str, int] = json.loads(file['atmname_to_idx'][()])
            self.scfx_weights: dict[str, float] = json.loads(file['scfx_weights'][()])
            self.ligand_sdf: str = file['ligand_sdf'].asstr()[()]
            self.mol = Chem.MolFromMolBlock( self.ligand_sdf, sanitize=False, removeHs=False )

            data = file['poses']['results']
            names = file['poses']['row_names'].asstr()[()]
            headers = file['poses']['results'].attrs['column_names']
            self.relax_df: pd.DataFrame = pd.DataFrame(data, columns=headers)
            self.relax_df.insert(loc=0, column='type', value=names)

            self.relax_pdb: list[str] = file['poses']['pdb_strings'].asstr()[()]

            data = file['results']
            names = file['protocol'].asstr()[()]
            headers = file['results'].attrs['column_names']
            self.docking_df: pd.DataFrame = pd.DataFrame(data, columns=headers)
            self.docking_df.insert(loc=0, column='type', value=names)

            self.docking_pdb_updates: list[dict[str, str]] = []
            for r in file['pdb_strings'][()]:
                self.docking_pdb_updates.append(json.loads(r))

    def relaxed_pdb(self, index: int) -> str:
        return self.relax_pdb[index]
    
    def docked_pdb(self, index: int) -> str:
        run_type = '_'.join(self.docking_df.iloc[index].type.split('_')[:-2])

        min_idx = self.relax_df[self.relax_df.type == run_type]['total_score'].idxmin()

        return pdb_result(self.relax_pdb[min_idx], self.docking_pdb_updates[index])
    
    def relaxed_mol(self, index: int) -> Chem.Mol:
        return mol_result(self.mol, self.relaxed_pdb(index), self.atmname_to_index)
    
    def docked_mol(self, index: int) -> Chem.Mol:
        return mol_result(self.mol, self.docked_pdb(index), self.atmname_to_index)
    
    def write_relaxed_mol(self, index: int, filename: str) -> None:
        with Chem.SDWriter(filename) as f:
            f.SetKekulize(False)
            f.write(self.relaxed_mol(index))
    
    def write_docked_mol(self, index: int, filename: str) -> None:
        with Chem.SDWriter(filename) as f:
            f.SetKekulize(False)
            f.write(self.docked_mol(index))

    def write_relax_pdb(self, index: int, filename: str, ligchain: chr = None) -> None:
        pdb_block = self.relaxed_pdb(index)
        if ligchain == None:
            with open(filename, 'w') as file:
                file.write(pdb_block)
        else:
            with open(filename, 'w') as file:
                for line in pdb_block.split('\n'):
                    if line[0:6] == 'HETATM' and line[17:20] == 'UNK':
                        line = line[:21] + ligchain + line[22:]
                    file.write(line + '\n')

    def write_docked_pdb(self, index: int, filename: str, ligchain: chr = None) -> None:
        pdb_block = self.docked_pdb(index)
        if ligchain == None:
            with open(filename, 'w') as file:
                file.write(pdb_block)
        else:
            with open(filename, 'w') as file:
                for line in pdb_block.split('\n'):
                    if line[0:6] == 'HETATM' and line[17:20] == 'UNK':
                        line = line[:21] + ligchain + line[22:]
                    file.write(line + '\n')

class PDBRinterface:

    def __init__(self, h5_directory: str ) -> None:

        self.directory: Path = Path(h5_directory)

        self.results: dict[str, str] = {}

        if self.directory.is_dir():
            for h5_file in self.directory.glob("*.h5"):
                self.results[h5_file.stem] = h5_file.absolute().as_posix()
        else:
            raise ValueError(h5_directory + " is not a directory. Please check provided path.")
        
        if len(self.results) == 0:
            raise ValueError("No h5 files found in " + h5_directory + ". Please check directory path.")
        
    def get_result(self, pdb_id: str) -> Result:
        return Result(self.results[pdb_id], pdb_id)
    

if __name__ == '__main__':

    results = PDBRinterface('h5_test')
    res = results.get_result('5k00')
    columns = ['total_score', 'idelta_score', 'rmsd_to_crystal', 'rmsd_to_input', 'prepare_time']
    print(res.relax_df[res.relax_df.type == 'relax'][columns])
    print(res.docking_df[res.docking_df.type == 'relax_docking_perturb'][columns])

    apo_relax = res.relax_df[res.relax_df.type == 'apo_relax'].iloc[0]
    relax = res.relax_df[res.relax_df.type == 'relax'].iloc[0]
    real_delta = {}
    for key in apo_relax.keys():
        if ('raw' in key and 'delta' not in key) or key == 'total_score':
            real_delta[key] = relax[key] - apo_relax[key]

    for key in real_delta.keys():
        if real_delta[key] != 0.0:
            print(key, real_delta[key], relax[key])

    from pyrosetta import *
    from load_ligand import pose_with_ligand

    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:300', silent=False)

    pose = pose_with_ligand(res.docked_pdb(0), res.docked_mol(0))

    xml_objects = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file('xml_protocols/docking_std.xml')
    scfx = xml_objects.get_score_function("hard_rep")

    print(scfx(pose), res.docking_df.iloc[0].total_score)
    print(scfx(pose)-res.docking_df.iloc[0].total_score)

    pose.dump_pdb("rosetta.pdb")
    res.write_docked_mol(0, 'test.sdf')
    res.write_docked_pdb(0, 'test.pdb')