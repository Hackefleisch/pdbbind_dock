from rdkit import Chem
from rdkit.Chem import AllChem

from pyrosetta import *

def generate_conformers(mol, nconf=0):
    mol_noH = Chem.RemoveHs( mol )
    nrotbonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol_noH, strict=True)

    # nconf logic adapted from Jean-Paul Ebejer's presentation
	# at the London RDKit User General Meeting
	# http://rdkit.org/UGM/2012/Ebejer_20110926_RDKit_1stUGM.pdf
    if nconf <= 0:
        nconf = 49
        if nrotbonds > 12:
            nconf = 299
        elif nrotbonds >= 8:
            nconf = 199
    else:
        nconf -= 1

    AllChem.EmbedMultipleConfs(mol, numConfs=nconf, clearConfs = False, maxAttempts = 30)
    AllChem.AlignMolConformers(mol)

    return mol


def load_ligand(lig_file):

    sdf_supplier = Chem.SDMolSupplier(lig_file, sanitize=True, removeHs=True)
    orig_mol = sdf_supplier[0]
    mol = Chem.AddHs(orig_mol)
    AllChem.ConstrainedEmbed(mol, orig_mol)

    return mol

def rdkit_to_mutable_res(mol):
    chem_manager = rosetta.core.chemical.ChemicalManager.get_instance()

    tag = "fa_standard"
    lig_restype = rosetta.core.chemical.MutableResidueType( 
        chem_manager.atom_type_set( tag ),
        chem_manager.element_set( "default" ),
        chem_manager.mm_atom_type_set( tag ),
        chem_manager.orbital_type_set( tag )
    )
    lig_restype.name( "UNKNOWN" )
    lig_restype.name3( "UNK" )
    lig_restype.name1( "X" )
    lig_restype.interchangeability_group( "UNK" )

    index_to_vd = {}

    conf = mol.GetConformer( 0 )

    for i in range( mol.GetNumAtoms() ):
        atom = mol.GetAtomWithIdx( i )
        element_name = atom.GetSymbol()
        charge = atom.GetFormalCharge()

        vd_atom = lig_restype.add_atom( "" )
        restype_atom = lig_restype.atom( vd_atom )
        restype_atom.element_type( lig_restype.element_set().element( element_name ) )
        restype_atom.formal_charge( charge )
        restype_atom.mm_name( "VIRT" )

        atom_pos = conf.GetAtomPosition( i )
        xyz = rosetta.numeric.xyzVector_double_t( atom_pos.x, atom_pos.y, atom_pos.z )
        restype_atom.ideal_xyz( xyz )

        index_to_vd[ i ] = vd_atom


    for bond in mol.GetBonds():
        bond_name = bond.GetBondType()
        if bond_name == Chem.rdchem.BondType.SINGLE:
            bond_name = rosetta.core.chemical.BondName.SingleBond
        elif bond_name == Chem.rdchem.BondType.DOUBLE:
            bond_name = rosetta.core.chemical.BondName.DoubleBond
        elif bond_name == Chem.rdchem.BondType.TRIPLE:
            bond_name = rosetta.core.chemical.BondName.TripleBond
        elif bond_name == Chem.rdchem.BondType.AROMATIC:
            bond_name = rosetta.core.chemical.BondName.AromaticBond
        else:
            print( "ERROR: encountered unknown bond type", bond_name )
            bond_name = rosetta.core.chemical.BondName.UnknownBond

        lig_restype.add_bond( 
            index_to_vd[ bond.GetBeginAtom().GetIdx() ],
            index_to_vd[ bond.GetEndAtom().GetIdx() ],
            bond_name
        )

    rosetta.core.chemical.rename_atoms( lig_restype, True )
    rosetta.core.chemical.rosetta_retype_fullatom( lig_restype, True )
    rosetta.core.chemical.rosetta_recharge_fullatom( lig_restype )

    rosetta.core.chemical.find_bonds_in_rings( lig_restype )

    nbr_vd = 0
    shortest_nbr_dist = 999999.99
    for vd in index_to_vd.values():
        if lig_restype.atom( vd ).element_type().get_chemical_symbol() == "H":
            continue
        tmp_dist = rosetta.core.chemical.find_nbr_dist( lig_restype, vd )
        if tmp_dist < shortest_nbr_dist:
            shortest_nbr_dist = tmp_dist
            nbr_vd = vd

    lig_restype.nbr_radius( shortest_nbr_dist )
    lig_restype.nbr_atom( nbr_vd )
    lig_restype.assign_internal_coordinates()
    lig_restype.autodetermine_chi_bonds()

    return lig_restype, index_to_vd

def add_confs_to_res(mol, mutable_res, index_to_vd):

    rotamers_spec = rosetta.core.chemical.rotamers.StoredRotamerLibrarySpecification()

    for i in range(mol.GetNumConformers()):
        conf = mol.GetConformer(i)
        single_conf_spec = pyrosetta.rosetta.std.map_std_string_numeric_xyzVector_double_t_std_allocator_std_pair_const_std_string_numeric_xyzVector_double_t()
        for idx, atm_vd in index_to_vd.items():
            rdkit_atm_pos = conf.GetAtomPosition(idx)
            single_conf_spec[ mutable_res.atom_name(atm_vd) ] = rosetta.numeric.xyzVector_double_t(rdkit_atm_pos.x, rdkit_atm_pos.y, rdkit_atm_pos.z)

        rotamers_spec.add_rotamer(single_conf_spec)

    mutable_res.rotamer_library_specification(rotamers_spec)
    return mutable_res
    

def mutable_res_to_res(mutable_res):
    lig_restype_non_mutable = rosetta.core.chemical.ResidueType.make( mutable_res )
    return rosetta.core.conformation.Residue( lig_restype_non_mutable, True )

def moltomolblock(mol):
    return Chem.MolToMolBlock(mol, kekulize = False, confId=0)

def molfrommolblock(molblock):
    return Chem.MolFromMolBlock(molblock, sanitize=False, removeHs=False)