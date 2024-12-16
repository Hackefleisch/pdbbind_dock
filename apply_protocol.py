from load_pose import load_pose

from pyrosetta import *

pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true ', silent=True)


def load_protocol(path):
    xml_objects = pyrosetta.rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file(path)
    return xml_objects.get_mover("ParsedProtocol")

def apply_protocol_to_pdbbind(path, pdb, n_runs, pdbbind='pdbbind_2020', clean=False):
    protocol = load_protocol(path)
    pose = load_pose(pdbbind, pdb, clean=clean)

    results = [pose]

    for i in range(n_runs):
        work_pose = pose.clone()
        protocol.apply(work_pose)
        results.append(work_pose)

    return results

pdb = '5zde'
poses = apply_protocol_to_pdbbind('xml_protocols/docking_std.xml', pdb, 10, clean=False)
for i in range(len(poses)):
    poses[i].dump_pdb('test_' + str(i) + '.pdb')