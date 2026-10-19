"""
docking.result -- reader for the per-complex docking HDF5 files (``Result`` / ``PDBRinterface``).

LEGACY SEMANTICS OF THE STORED DATA (all files; the docking code is unchanged and no re-docking
is planned, so these hold for every file under data/*_h5/)
-------------------------------------------------------------------------------------------
* ``total_score`` as stored INCLUDES the coordinate-constraint energy
  (``scfx_weights['coordinate_constraint']`` x ``raw_coordinate_constraint``): the runner scored
  the relaxed seeds before removing the FastRelax restraints, and the FinalMinimizer of the
  docking protocol leaves its C-alpha restraints in the pose. ``idelta_score`` is not affected.
* ``rmsd_to_crystal`` / ``rmsd_to_input`` are Rosetta ``RMSDMetric`` values: in place (no
  superposition) over all heavy atoms, matched by atom name, i.e. NOT symmetry-corrected.

AUTO-CORRECTION (``autocorrect=True``, the default)
---------------------------------------------------
``Result`` corrects exactly these three columns on access and keeps the stored values as
``total_score_legacy``, ``rmsd_to_crystal_legacy`` and ``rmsd_to_input_legacy``:

* ``total_score``  <- stored total minus the weighted coordinate-constraint energy;
* ``rmsd_to_crystal`` / ``rmsd_to_input``  <- symmetry-corrected, in-place heavy-atom RMSDs
  (``docking.ligand_rmsd``, validated against Rosetta's ``automorphic_rmsd``), recomputed from
  the stored structures. For the seed rows ``rmsd_to_input`` becomes the RMSD to the crystal
  ligand (the input of the relaxation); the stored value there is 0.

Everything else is left exactly as stored. Pass ``autocorrect=False`` for the raw legacy
tables. The RMSD correction needs the stored structures (about one second per file). The
docking reference of each receptor state is always chosen with the *stored* total score,
because that is what the docking run used (``reference_index``).
"""

from pathlib import Path
import h5py
import json
import numpy as np
import pandas as pd
from rdkit import Chem
import copy
import zlib

from docking.load_ligand import mol_result

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

    #: columns rewritten by the auto-correction; the stored values move to ``<column>_legacy``
    CORRECTED_COLUMNS = ('total_score', 'rmsd_to_crystal', 'rmsd_to_input')

    def __init__(self, h5_file_path: str, name: str, load_structure: bool = True,
                 autocorrect: bool = True) -> None:
        self.name: str = name
        self.path: str = h5_file_path
        self.autocorrect: bool = autocorrect
        self.ligand_symmetry = None          # docking.ligand_rmsd.LigandSymmetry once computed
        self.n_rmsd_uncorrected: int = 0     # rows whose RMSD could not be recomputed (legacy kept)
        self.autocorrect_notes: list[str] = []

        with h5py.File(self.path, 'r') as file:
            data = file['poses']['results']
            names = file['poses']['row_names'].asstr()[()]
            headers = file['poses']['results'].attrs['column_names']
            self.relax_df: pd.DataFrame = pd.DataFrame(data, columns=headers)
            self.relax_df.insert(loc=0, column='type', value=names)

            data = file['results']
            names = file['protocol'].asstr()[()]
            headers = file['results'].attrs['column_names']
            self.docking_df: pd.DataFrame = pd.DataFrame(data, columns=headers)
            self.docking_df.insert(loc=0, column='type', value=names)

            self.scfx_weights: dict[str, float] = json.loads(file['scfx_weights'][()])

            if load_structure or autocorrect:
                self.atmname_to_index: dict[str, int] = json.loads(file['atmname_to_idx'][()])
                self.ligand_sdf: str = file['ligand_sdf'].asstr()[()]
                self.mol = Chem.MolFromMolBlock( self.ligand_sdf, sanitize=False, removeHs=False )

                self.relax_pdb: list[str] = []
                for blob in file['poses']['pdb_strings']:
                    # blob (uint8 array) -> bytes -> decompress -> string
                    pdb_str = zlib.decompress(blob.tobytes()).decode('utf-8')
                    self.relax_pdb.append(pdb_str)

                self.docking_pdb_updates: list[dict[str, str]] = []
                for blob in file['pdb_strings']:
                    # blob (uint8 array) -> bytes -> decompress -> string -> json
                    json_str = zlib.decompress(blob.tobytes()).decode('utf-8')
                    self.docking_pdb_updates.append(json.loads(json_str))

        if autocorrect:
            self._autocorrect_total_score()
            self._autocorrect_rmsd()
            if not load_structure:
                # keep the memory footprint of load_structure=False
                del self.relax_pdb
                del self.docking_pdb_updates

    # ------------------------------------------------------------------ #
    # auto-correction
    # ------------------------------------------------------------------ #
    def _autocorrect_total_score(self) -> None:
        """total_score <- stored total minus the weighted coordinate-constraint energy."""
        w = float(self.scfx_weights.get('coordinate_constraint', 0.0))
        for df in (self.relax_df, self.docking_df):
            if 'total_score_legacy' in df.columns:
                continue
            df['total_score_legacy'] = df['total_score'].copy()
            if w != 0.0 and 'raw_coordinate_constraint' in df.columns:
                df['total_score'] = (df['total_score_legacy'] - w * df['raw_coordinate_constraint']).astype(np.float32)
        self.autocorrect_notes.append(f'total_score: subtracted {w} x raw_coordinate_constraint')

    def _autocorrect_rmsd(self) -> None:
        """rmsd_to_crystal / rmsd_to_input <- symmetry-corrected in-place RMSDs from the stored structures."""
        from docking.ligand_rmsd import (LigandSymmetry, docked_ligand_lines,
                                         heavy_xyz_from_pdb_lines, ligand_lines)

        for df in (self.relax_df, self.docking_df):
            for col in ('rmsd_to_crystal', 'rmsd_to_input'):
                if f'{col}_legacy' not in df.columns:
                    df[f'{col}_legacy'] = df[col].copy()

        sym = LigandSymmetry(self.mol)
        self.ligand_symmetry = sym

        seed_lines = [s.split('\n') for s in self.relax_pdb]
        seed_xyz: list = []
        for lines in seed_lines:
            try:
                li = ligand_lines(lines)
                seed_xyz.append(heavy_xyz_from_pdb_lines([lines[i] for i in li], self.atmname_to_index, sym))
            except Exception:
                seed_xyz.append(None)

        crystal_rows = np.where(self.relax_df['type'].to_numpy() == 'crystal')[0]
        crystal_xyz = seed_xyz[crystal_rows[0]] if len(crystal_rows) else None
        if crystal_xyz is None:
            self.n_rmsd_uncorrected = len(self.relax_df) + len(self.docking_df)
            self.autocorrect_notes.append('rmsd: crystal ligand not available, legacy values kept')
            return

        # seeds: RMSD to the crystal ligand (also the input of the relaxation)
        vals = self.relax_df['rmsd_to_crystal_legacy'].to_numpy(dtype=np.float32).copy()
        for k, xyz in enumerate(seed_xyz):
            if xyz is not None:
                vals[k] = sym.rmsd(xyz, crystal_xyz)
            else:
                self.n_rmsd_uncorrected += 1
        self.relax_df['rmsd_to_crystal'] = vals
        self.relax_df['rmsd_to_input'] = vals.copy()

        # docked poses: ligand lines collected from the diff against the docking reference seed
        protocols = self.docking_df['type'].to_numpy()
        states = np.array(['_'.join(p.split('_')[:-2]) for p in protocols])
        ref = {}
        for st in np.unique(states):
            r = self.reference_index(st)
            ref[st] = (seed_lines[r], ligand_lines(seed_lines[r]), seed_xyz[r])
        n = len(protocols)
        xyz_all = np.full((n, sym.n_heavy, 3), np.nan)
        for k, upd in enumerate(self.docking_pdb_updates[:n]):
            lines, li, _ = ref[states[k]]
            try:
                xyz_all[k] = heavy_xyz_from_pdb_lines(docked_ligand_lines(lines, li, upd), self.atmname_to_index, sym)
            except Exception:
                pass
        ok = ~np.isnan(xyz_all).any(axis=(1, 2))
        rc = self.docking_df['rmsd_to_crystal_legacy'].to_numpy(dtype=np.float32).copy()
        ri = self.docking_df['rmsd_to_input_legacy'].to_numpy(dtype=np.float32).copy()
        if ok.any():
            rc[ok] = sym.rmsd_many(xyz_all[ok], crystal_xyz)
            for st, (_, _, rxyz) in ref.items():
                m = ok & (states == st)
                if m.any() and rxyz is not None:
                    ri[m] = sym.rmsd_many(xyz_all[m], rxyz)
        self.n_rmsd_uncorrected += int((~ok).sum())
        self.docking_df['rmsd_to_crystal'] = rc
        self.docking_df['rmsd_to_input'] = ri
        self.autocorrect_notes.append(
            f'rmsd: symmetry-corrected in place ({sym.n_automorphisms} automorphisms'
            f'{", capped" if sym.capped else ""}; {self.n_rmsd_uncorrected} rows kept legacy)')

    def reference_index(self, state: str) -> int:
        """Row of ``relax_df`` that the docking of *state* started from: the lowest **stored**
        total score of that state (``runner.run()`` selected it before any correction)."""
        col = 'total_score_legacy' if 'total_score_legacy' in self.relax_df.columns else 'total_score'
        return int(self.relax_df[self.relax_df.type == state][col].idxmin())

    # ------------------------------------------------------------------ #
    # structures
    # ------------------------------------------------------------------ #
    def relaxed_pdb(self, index: int) -> str:
        return self.relax_pdb[index]
    
    def docked_pdb(self, index: int) -> str:
        run_type = '_'.join(self.docking_df.iloc[index].type.split('_')[:-2])

        min_idx = self.reference_index(run_type)

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
        
    def get_result(self, pdb_id: str, load_structure: bool = True, autocorrect: bool = True) -> Result:
        return Result(self.results[pdb_id], pdb_id, load_structure, autocorrect)
    

if __name__ == '__main__':

    results = PDBRinterface('data/h5_test')
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
    from docking.load_ligand import pose_with_ligand

    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:300', silent=False)

    pose = pose_with_ligand(res.docked_pdb(0), res.docked_mol(0))

    xml_objects = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file('docking/xml_protocols/docking_std.xml')
    scfx = xml_objects.get_score_function("hard_rep")

    print(scfx(pose), res.docking_df.iloc[0].total_score)
    print(scfx(pose)-res.docking_df.iloc[0].total_score)

    pose.dump_pdb("rosetta.pdb")
    res.write_docked_mol(0, 'test.sdf')
    res.write_docked_pdb(0, 'test.pdb')
