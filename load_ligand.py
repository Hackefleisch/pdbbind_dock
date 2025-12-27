from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import AllChem

from rdkit.Geometry import Point3D

from pyrosetta import *

def generate_conformers(mol, nconf=0, seed=-1):
    try:
        mol_noH = Chem.RemoveHs( mol )
        nrotbonds = Lipinski.NumRotatableBonds(mol_noH)
    except:
        print("Ligand is most likely unable to be kekulized. Generating 100 conformers as default")
        nconf = 100
    
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
        # since the initial conformation will be kept
        nconf -= 1


    embed_params = AllChem.ETKDGv3()
    embed_params.clearConfs = False
    embed_params.randomSeed = seed
    AllChem.EmbedMultipleConfs(mol, nconf, embed_params)
    AllChem.AlignMolConformers(mol)

    return mol


def load_ligand(lig_file):

    filetype = lig_file.split('.')[-1]
    if filetype == "sdf":
        sdf_supplier = Chem.SDMolSupplier(lig_file, sanitize=False, removeHs=False)
        orig_mol = sdf_supplier[0]
    elif filetype == "mol2":
        orig_mol = Chem.MolFromMol2File(lig_file, sanitize=False, removeHs=False)
    #mol = Chem.AddHs(orig_mol)
    #AllChem.ConstrainedEmbed(mol, orig_mol)

    mol = reprotonate_mol(orig_mol)

    return mol

def reprotonate_mol(mol):
    """
    Reprotonates an RDKit molecule using logic ported from Rosetta's C++ reprotonate_rdmol written by Rocco Moretti.
    Preserves 3D coordinates of heavy atoms and generates coordinates for new Hydrogens.
    
    Args:
        mol (rdkit.Chem.Mol): The input molecule (with 3D coords).
        
    Returns:
        rdkit.Chem.Mol: The reprotonated molecule.
    """
    
    # 1. Remove hydrogens if needed
    mol = Chem.RemoveHs(mol, sanitize=False)

    # 2. Soft Sanitize
    mol.ClearComputedProps()
    Chem.Cleanup(mol)
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol) # Computes ring info (symmetrizeSSSR)

    # 3. Remove Excess Protons
    # Removes hydrogens that contribute to a positive formal charge
    for atom in mol.GetAtoms():
        charge = atom.GetFormalCharge()
        if charge > 0:
            # Check explicit Hs
            n_explicit = atom.GetNumExplicitHs()
            if n_explicit > 0:
                delta = min(charge, n_explicit)
                atom.SetNumExplicitHs(n_explicit - delta)
                atom.SetFormalCharge(charge - delta)
                atom.UpdatePropertyCache(strict=False) # Reset implicit/explicit counts
            
            # Re-check charge after explicit adjustment to handle residual implicits
            charge = atom.GetFormalCharge()
            n_implicit = atom.GetNumImplicitHs()
            if charge > 0 and n_implicit > 0:
                delta = min(charge, n_implicit)
                atom.SetFormalCharge(charge - delta)
                atom.UpdatePropertyCache(strict=False)

    # 4. Apply Charge Transforms
    
    # Define the transforms (SMARTS pattern, Target Charge)
    # Order matches the C++ static const PH_TRANSFORMS list
    transforms = [
        # Singly connected or Doubly connected negative oxygen
        ("[O-1$([OD1]),$([OD2])]", 0),
        # Negative sulfur
        ("[S-1$([SD1]),$([SD2])]", 0),
        # Fix up: Oxygen next to double bonded O/S
        ("[O-0D1$(O-[*]=[O,S,o,s])]", -1),
        # Fix up: Sulfur next to double bonded O/S
        ("[S-0D1$(S-[*]=[O,S,o,s])]", -1),
        # Nitrogen fixes (complicated specificity to avoid specific conjugated systems)
        ("[N+0&!$(N=[*])&!$(N#[*])&!$(N-[*]=[*])&!$(N-[*]#[*])&!$(N-[*]:[*])]", 1),
        # Amidine & Guanidine
        ("[N+0D1$(N=C-[N+0])]", 1),
        # Nitro group ylide representation
        ("[O-0D1H1$(O-[N+]=O)]", -1)
    ]

    for smarts, target_charge in transforms:
        pattern = Chem.MolFromSmarts(smarts)
        if not pattern:
            continue
            
        # Get all matches. Uniquify=True is default.
        matches = mol.GetSubstructMatches(pattern)
        
        for match in matches:
            # Assume the first atom in the match is the target (index 0)
            atom = mol.GetAtomWithIdx(match[0])
            
            current_charge = atom.GetFormalCharge()
            
            # Remove explicit hydrogens if charge is being reduced
            if target_charge < current_charge:
                charge_delta = current_charge - target_charge
                explicit_hs = atom.GetNumExplicitHs()
                
                if charge_delta < explicit_hs:
                    atom.SetNumExplicitHs(explicit_hs - charge_delta)
                else:
                    atom.SetNumExplicitHs(0)
            
            atom.SetNoImplicit(False)
            atom.SetFormalCharge(target_charge)
            atom.UpdatePropertyCache(strict=False)

    # 5. Final Sanitize and Add Hs
    # This recalculates valence, aromaticity, conjugation, etc.
    Chem.SanitizeMol(mol)
    
    # C++: addHs(rdmol, false, /*addCoords=*/ true)
    # addCoords=True ensures 3D coordinates are generated for the new Hs
    # based on the existing heavy atom geometry.
    mol = Chem.AddHs(mol, addCoords=True)
    
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

def pose_with_ligand(pdb_string, rdkit_mol, mut_res=None, index_to_vd=None):
    if mut_res == None or index_to_vd == None:
        mut_res, index_to_vd = rdkit_to_mutable_res(rdkit_mol)
    mut_res = add_confs_to_res(rdkit_mol, mut_res, index_to_vd)
    res = mutable_res_to_res(mut_res)

    pose = rosetta.core.pose.Pose()
    rosetta.core.import_pose.pose_from_pdbstring(pose, pdb_string)

    pose.append_residue_by_jump( res, 1, "", "", True )
    pose.pdb_info().chain( pose.total_residue(), 'X' )
    pose.update_pose_chains_from_pdb_chains()

    return pose

def mol_result( rdkit_mol, pdb_str, atmname_to_index ):
    """
    This function uses the sanitized rdkit sdf and a pdb to extract a ligand conformer and add it to the rdkit mol object.
    """

    for line in pdb_str.split('\n'):
        if line[:6] == 'HETATM' and line[17:20] == 'UNK':
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            name = line[12:16]
            rdkit_mol.GetConformer(0).SetAtomPosition(atmname_to_index[name], Point3D(x,y,z))

    return rdkit_mol

if __name__ == '__main__':

    mol = load_ligand(lig_file='pdbbind_cleaned/10gs/10gs_ligand.sdf')

    mol = generate_conformers(mol)

    output_filename = "reprotonated.sdf"
    writer = Chem.SDWriter(output_filename)
    writer.SetKekulize(False)
    for cid in range(mol.GetNumConformers()):
        writer.write(mol, confId=cid)
    writer.close()

    mol = load_ligand('reprotonated.sdf')
    print(mol)