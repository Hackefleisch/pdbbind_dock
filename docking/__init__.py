"""
docking
=======
Phase-1 docking engine (PyRosetta).

Modules
-------
- runner       : the docking engine (relax + dock, writes one HDF5 per complex)
- main         : CLI entry point (one complex per invocation; for SLURM arrays)
- load_ligand  : RDKit <-> Rosetta ligand handling
- result       : reader for the docking HDF5 files (``Result`` / ``PDBRinterface``)

Run the docking CLI as a module from the repo root::

    python -m docking.main --help
"""
