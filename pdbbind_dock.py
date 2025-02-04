# This is a helper script to easily access docking results

import zarr
from rdkit import Chem
from rdkit.Geometry import Point3D
import copy

def pdb_result( orig_pdb_strarr, update_pdb_strarr ):
        
    result = copy.deepcopy( orig_pdb_strarr )
    for idx, upd in update_pdb_strarr.items():
        result[int(idx)] = upd
    return "\n".join(result)
    
def mol_result( rdkit_mol, pdb_str, atmname_to_index ):

    for line in pdb_str.split('\n'):
        if line[:6] == 'HETATM' and line[17:20] == 'UNK':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            name = line[12:16]
            rdkit_mol.GetConformer().SetAtomPosition(atmname_to_index[name], Point3D(x,y,z))

    return rdkit_mol

class PDBrun:

    def __init__(self, result_dict, atmname_to_index, native_molblock, input_pdb):
        
        self.pdb = pdb_result( input_pdb, result_dict['pdb_string_delta'] )
        mol = Chem.MolFromMolBlock( native_molblock, sanitize=False, removeHs=False )
        self.rdkit_mol = mol_result( mol, self.pdb, atmname_to_index )

        self.total_score = result_dict[ 'total_score' ]
        self.idelta_score = result_dict[ 'idelta_score' ]
        self.raw_energies = result_dict[ 'raw_energies' ]
        self.raw_delta_energies = result_dict[ 'raw_delta_energies' ]

        self.rmsd_to_crystal = result_dict[ 'rmsd_to_crystal' ]
        self.rmsd_to_input = result_dict[ 'rmsd_to_input' ]
        self.rmsd_to_lowest_idelta = result_dict[ 'rmsd_to_lowest_idelta' ]
        self.rmsd_to_lowest_score = result_dict[ 'rmsd_to_lowest_score' ]
        
    def write_mol(self, filename):
        with Chem.SDWriter(filename) as f:
            f.SetKekulize(False)
            f.write(self.rdkit_mol)

    def write_pdb(self, filename):
        with open(filename, 'w') as file:
            file.write(self.pdb)

class PDBcomplex:

    def __init__(self, result_dict, atmname_to_index, native_molblock ):
        
        self.pdb = pdb_result( result_dict[ 'pdb_string_arr' ], {} )
        mol = Chem.MolFromMolBlock( native_molblock, sanitize=False, removeHs=False )
        self.rdkit_mol = mol_result( mol, self.pdb, atmname_to_index )


        self.total_score = result_dict[ 'total_score' ]
        self.idelta_score = result_dict[ 'idelta_score' ]
        self.raw_energies = result_dict[ 'raw_energies' ]
        self.raw_delta_energies = result_dict[ 'raw_delta_energies' ]

        self.prepare_time = result_dict[ 'prepare_time' ]
        self.rmsd_to_crystal = result_dict[ 'rmsd_to_crystal' ]
        self.score_distribution = result_dict[ 'score_distribution' ]
        
    def write_mol(self, filename):
        with Chem.SDWriter(filename) as f:
            f.SetKekulize(False)
            f.write(self.rdkit_mol)

    def write_pdb(self, filename):
        with open(filename, 'w') as file:
            file.write(self.pdb)

class PDBresult:

    def __init__(self, zarr_store, pdb_id):
        
        self.results = zarr_store[pdb_id][pdb_id][0]
        self.pdb_id = pdb_id
        self.sanitized_molblock = "\n".join( self.results[ 'sanitized_sdf' ] )
        self.atmname_to_index = self.results[ 'atmname_to_idx' ]

    def get_complex(self, label):
        if label not in self.results[ 'complex_results' ]:
            raise KeyError( label + " not found in complex_results for " + self.pdb_id + ". Available keys: " + ", ".join(list( self.results[ 'complex_results' ].keys()) ) )
        return PDBcomplex( self.results[ 'complex_results' ][ label ], self.atmname_to_index, self.sanitized_molblock )

    def get_run(self, label, index):
        if label not in self.results[ 'docking_results' ]:
            raise KeyError( label + " not found in docking_results for " + self.pdb_id + ". Available keys: " + ", ".join(list( self.results[ 'docking_results' ].keys()) ) )
        if index < 0 or index >= len( self.results[ 'docking_results' ][ label ] ):
            raise KeyError( "Index " + str(index) + " is out of bound for " + self.pdb_id + " protocol " + label + ": [0," + str(len( self.results[ 'docking_results' ][ label ] ) ) + "]" )
        input_pdb_structure = self.results[ 'docking_results' ][ label ][ index ][ 'input_pose_name' ]
        return PDBrun( self.results[ 'docking_results' ][ label ][ index ], self.atmname_to_index, self.sanitized_molblock, self.results[ 'complex_results' ][ input_pdb_structure ][ 'pdb_string_arr' ] )

store = zarr.open('formatted_results.zarr', 'r')
pdb_entry = PDBresult( store, '184l' )
pose_relax = pdb_entry.get_complex('pose_relax')
print(pose_relax.idelta_score)

perturb_relax_4 = pdb_entry.get_run( 'docking_perturb_pose_relax', 4 )
print(perturb_relax_4.idelta_score)

pose_relax_ligaway = pdb_entry.get_complex( 'pose_relax_ligaway' )
real_delta = {}
for key in pose_relax_ligaway.raw_energies.keys():
    real_delta[key] = pose_relax.raw_energies[key] - pose_relax_ligaway.raw_energies[key]

for key in real_delta.keys():
    if real_delta[key] != 0.0:
        print(key, real_delta[key], pose_relax.raw_delta_energies[key])

"""
from pyrosetta import *
from load_ligand import rdkit_to_mutable_res, mutable_res_to_res

pyrosetta.init(options='-in:auto_setup_metals -ex1 -ex2 -restore_pre_talaris_2013_behavior true -out:levels all:300', silent=False)

pose = rosetta.core.pose.Pose()
rosetta.core.import_pose.pose_from_pdbstring(pose, pr.pdb)

mut_res, _ = rdkit_to_mutable_res(pr.rdkit_mol)
res = mutable_res_to_res(mut_res)

pose.append_residue_by_jump( res, 1, "", "", True )
pose.pdb_info().chain( pose.total_residue(), 'X' )
pose.update_pose_chains_from_pdb_chains()

xml_objects = rosetta.protocols.rosetta_scripts.XmlObjects.create_from_file('xml_protocols/docking_std.xml')
scfx = xml_objects.get_score_function("hard_rep")

print(scfx(pose), pr.total_score)

pose.dump_pdb("rosetta.pdb")
pr.write_pdb("test.pdb")
"""