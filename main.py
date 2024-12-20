from pyrosetta import *

from Runner import Runner
import pickle



def main(pdbbind_path, pdb):
    r = Runner(pdbbind_path, pdb, ['xml_protocols/docking_std.xml', 'xml_protocols/docking_perturb.xml'])
    r.run(1, 0)

    features = {
        "pdb" : r.pdb,
        "conformers" : r.conformers,
        "atm_to_idx" : r.atmname_to_idx,
        "input_complexes" : r.complex_results,
        "docking_results" : r.docking_results,
    }



    with open("run.pickle", 'wb') as f:
        pickle.dump(features, f)

    #print(r.complex_results)



if __name__ == '__main__':
    import argparse

    pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:level 100', silent=True)

    main( 'pdbbind_2020', '4f4p' )