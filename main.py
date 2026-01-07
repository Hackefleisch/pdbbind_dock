from pyrosetta import *

from Runner import Runner
import pathlib
import shutil


def main(pdbbind_path, pdb_file, pdb_index, n_relax, n_relax_ligaway, n_dock, outdir, protocols):

    print("Loading pdb codes...", flush=True)

    # load pdbs
    pdb = ""
    counter = 0
    with open(pdb_file) as f:
        for line in f:
            # skip comments
            if line[0] == '#':
                continue
            pdb_id = line[:4]
            if counter == pdb_index:
                pdb = pdb_id
                break
            counter +=1

    print("Process", pdb_index, "selected complex", pdb, flush=True)

    try:
        r = Runner(
            pdbbind_path=pdbbind_path.as_posix(),
            pdb=pdb,
            protocol_paths=[p.as_posix() for p in protocols],
            output_dir=outdir.as_posix()
        )
        r.run(
            n_relax=n_relax,
            n_apo_relax=n_relax_ligaway,
            n_dock=n_dock
        )
    finally:
        r.close()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                    prog='PDBBindDocker',
                    description='This program can dock a given pdb structure.',
                    epilog='Lorem Ipsum')
    
    parser.add_argument( "--n_relax", default=10, type=int, help='How often should the pose with ligand be relaxed. The lowest scoring pose will be used for docking.' )
    parser.add_argument( "--n_relax_ligaway", default=10, type=int, help='How often should the pose without ligand be relaxed. The lowest scoring pose will be used for docking.' )
    parser.add_argument( "--n_dock", default=150, type=int, help='How often should each docking protocol be applied to each pose.' )
    parser.add_argument( "--outdir", type=pathlib.Path, help="Location of a directory to store hd5 files in", required=True )
    parser.add_argument( "--pdbbind", type=pathlib.Path, help="Location of pdbbind directory", required=True )
    parser.add_argument( "--pdb_file", type=pathlib.Path, help="Location of pdb list to dock", required=True )
    parser.add_argument( "--pdb_index", type=int, help='Line index in pdbfile', required=True )
    parser.add_argument( "--protocols", nargs='+', type=pathlib.Path, help='Provide multiple Rosetta protocols to execute for docking', required=True)
    parser.add_argument( "--overwrite", action='store_true', help='If set the specified zarr store will be cleared before running. Use with caution!')

    args = parser.parse_args()

    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:100', silent=True)

    if args.overwrite and os.path.exists(args.zarr_store):
        shutil.rmtree(args.zarr_store)

    main( 
        pdbbind_path=args.pdbbind,
        pdb_file=args.pdb_file,
        pdb_index=args.pdb_index,
        n_relax=args.n_relax, 
        n_relax_ligaway=args.n_relax_ligaway, 
        n_dock=args.n_dock, 
        outdir=args.outdir,
        protocols=args.protocols,
    )