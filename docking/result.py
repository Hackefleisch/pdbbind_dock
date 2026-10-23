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
from rdkit.Geometry import Point3D
import copy
import zlib


def mol_result(rdkit_mol, pdb_str, atmname_to_index):
    """Copy of the ligand whose conformer 0 carries the coordinates of the ``UNK`` HETATM lines of
    *pdb_str*; the original conformer 0 is appended as the last conformer.

    Same semantics as ``docking.load_ligand.mol_result``, duplicated here so that reading the HDF5
    files (and therefore the whole analysis) does not import PyRosetta. ``docking.load_ligand`` stays
    unchanged because it is part of the docking engine that produced the data.
    """
    mol = copy.deepcopy(rdkit_mol)
    conf = Chem.Conformer(mol.GetConformer(0))
    mol.AddConformer(conf, mol.GetNumConformers())
    for line in pdb_str.split('\n'):
        if line[:6] == 'HETATM' and line[17:20] == 'UNK':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            name = line[12:16]
            mol.GetConformer(0).SetAtomPosition(atmname_to_index[name], Point3D(x, y, z))
    return mol


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
        self.ligand_symmetry = None          # docking.ligand_rmsd.LigandSymmetry once computed (``symmetry``)
        self._seed_xyz = None                # cache of seed_ligand_xyz()
        self._docked_xyz = None              # cache of docked_ligand_xyz()
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
                self._seed_xyz = self._docked_xyz = None

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
        for df in (self.relax_df, self.docking_df):
            for col in ('rmsd_to_crystal', 'rmsd_to_input'):
                if f'{col}_legacy' not in df.columns:
                    df[f'{col}_legacy'] = df[col].copy()

        sym = self.symmetry
        seed_xyz = self.seed_ligand_xyz()
        seed_ok = ~np.isnan(seed_xyz).any(axis=(1, 2))
        crystal_rows = np.where(self.relax_df['type'].to_numpy() == 'crystal')[0]
        if len(crystal_rows) == 0 or not seed_ok[crystal_rows[0]]:
            self.n_rmsd_uncorrected = len(self.relax_df) + len(self.docking_df)
            self.autocorrect_notes.append('rmsd: crystal ligand not available, legacy values kept')
            return
        crystal_xyz = seed_xyz[crystal_rows[0]]

        # seeds: RMSD to the crystal ligand (also the input of the relaxation)
        vals = self.relax_df['rmsd_to_crystal_legacy'].to_numpy(dtype=np.float32).copy()
        vals[seed_ok] = sym.rmsd_many(seed_xyz[seed_ok], crystal_xyz)
        self.n_rmsd_uncorrected += int((~seed_ok).sum())
        self.relax_df['rmsd_to_crystal'] = vals
        self.relax_df['rmsd_to_input'] = vals.copy()

        # docked poses: ligand coordinates reconstructed from the diff against the docking reference seed
        xyz_all = self.docked_ligand_xyz()
        protocols = self.docking_df['type'].to_numpy()
        states = np.array(['_'.join(p.split('_')[:-2]) for p in protocols])
        ok = ~np.isnan(xyz_all).any(axis=(1, 2))
        rc = self.docking_df['rmsd_to_crystal_legacy'].to_numpy(dtype=np.float32).copy()
        ri = self.docking_df['rmsd_to_input_legacy'].to_numpy(dtype=np.float32).copy()
        if ok.any():
            rc[ok] = sym.rmsd_many(xyz_all[ok], crystal_xyz)
            for st in np.unique(states):
                r = self.reference_index(st)
                m = ok & (states == st)
                if m.any() and seed_ok[r]:
                    ri[m] = sym.rmsd_many(xyz_all[m], seed_xyz[r])
        self.n_rmsd_uncorrected += int((~ok).sum())
        self.docking_df['rmsd_to_crystal'] = rc
        self.docking_df['rmsd_to_input'] = ri
        self.autocorrect_notes.append(
            f'rmsd: symmetry-corrected in place ({sym.n_automorphisms} automorphisms'
            f'{", capped" if sym.capped else ""}; {self.n_rmsd_uncorrected} rows kept legacy)')

    # ------------------------------------------------------------------ #
    # ligand coordinates (used by the auto-correction and by affinity.basins)
    # ------------------------------------------------------------------ #
    @property
    def symmetry(self):
        """``docking.ligand_rmsd.LigandSymmetry`` of the ligand (heavy-atom automorphisms), computed once."""
        if self.ligand_symmetry is None:
            from docking.ligand_rmsd import LigandSymmetry
            self.ligand_symmetry = LigandSymmetry(self.mol)
        return self.ligand_symmetry

    def _need_structures(self) -> None:
        if not hasattr(self, 'relax_pdb'):
            raise RuntimeError('the stored structures are not loaded (Result(..., load_structure=False))')

    def seed_ligand_xyz(self) -> np.ndarray:
        """Heavy-atom coordinates of the ligand of every seed row, shape (n_seed, n_heavy, 3) in
        ``relax_df`` order, atoms in ``symmetry.heavy_idx`` order; NaN rows where the ligand could
        not be read. Needs the stored structures (``load_structure=True``)."""
        if self._seed_xyz is None:
            from docking.ligand_rmsd import heavy_xyz_from_pdb_lines, ligand_lines
            self._need_structures()
            sym = self.symmetry
            out = np.full((len(self.relax_pdb), sym.n_heavy, 3), np.nan)
            for k, pdb_str in enumerate(self.relax_pdb):
                lines = pdb_str.split('\n')
                try:
                    li = ligand_lines(lines)
                    out[k] = heavy_xyz_from_pdb_lines([lines[i] for i in li], self.atmname_to_index, sym)
                except Exception:
                    pass
            self._seed_xyz = out
        return self._seed_xyz

    def docked_ligand_xyz(self) -> np.ndarray:
        """Heavy-atom coordinates of the ligand of every docked pose, shape (n_docked, n_heavy, 3)
        in ``docking_df`` order, atoms in ``symmetry.heavy_idx`` order, reconstructed from the
        stored line diffs against the docking reference seed of the pose's receptor state
        (``reference_index``); NaN rows where the ligand could not be reconstructed. Needs the
        stored structures (``load_structure=True``)."""
        if self._docked_xyz is None:
            from docking.ligand_rmsd import docked_ligand_lines, heavy_xyz_from_pdb_lines, ligand_lines
            self._need_structures()
            sym = self.symmetry
            protocols = self.docking_df['type'].to_numpy()
            states = np.array(['_'.join(p.split('_')[:-2]) for p in protocols])
            ref = {}
            for st in np.unique(states):
                lines = self.relax_pdb[self.reference_index(st)].split('\n')
                ref[st] = (lines, ligand_lines(lines))
            n = len(protocols)
            out = np.full((n, sym.n_heavy, 3), np.nan)
            for k, upd in enumerate(self.docking_pdb_updates[:n]):
                lines, li = ref[states[k]]
                try:
                    out[k] = heavy_xyz_from_pdb_lines(docked_ligand_lines(lines, li, upd), self.atmname_to_index, sym)
                except Exception:
                    pass
            self._docked_xyz = out
        return self._docked_xyz

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
