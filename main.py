from pyrosetta import *

from Runner import Runner
import pickle



def main(pdbbind_path, pdb, n_relax, n_relax_ligaway, n_runs):
    r = Runner(pdbbind_path, pdb, ['xml_protocols/docking_std.xml', 'xml_protocols/docking_perturb.xml'], zarr_path='test.zarr')
    r.run(n_relax=n_relax, n_relax_ligaway=n_relax_ligaway, n_runs=n_runs)
    features = {
        "pdb" : r.pdb,
        #"conformers" : r.conformers,
        "atm_to_idx" : r.atmname_to_idx,
        "input_complexes" : r.complex_results,
        "docking_results" : r.docking_results,
    }



    with open("run.pickle", 'wb') as f:
        pickle.dump(features, f)

    #print(r.complex_results)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                    prog='PDBBindDocker',
                    description='This program can dock a given pdb structure.',
                    epilog='Lorem Ipsum')
    
    parser.add_argument( "--n_relax", default=10, type=int, help='How often should the pose with ligand be relaxed. The lowest scoring pose will be used for docking.' )
    parser.add_argument( "--n_relax_ligaway", default=10, type=int, help='How often should the pose without ligand be relaxed. The lowest scoring pose will be used for docking.' )
    parser.add_argument( "--n_dock", default=150, type=int, help='How often should each docking protocol be applied to each pose.' )

    args = parser.parse_args()

    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:100', silent=True)

    main( 'pdbbind_2020', '4f4p', args.n_relax, args.n_relax_ligaway, args.n_dock )