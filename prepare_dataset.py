# Pandas dataframe features
# pdbcode,lppdbbind_split,affinity_type,dG,vague_measures,covalent,fully_relaxed,nres,qed,has_rare_element
#
# crl: crystal, relaxed, ligaway
# rosetta values: relax rmsd to crystal, crl total score, cr idelta, cr raw delta energies, rl raw energies
#
# Additionally: Safe pdb files and sdf (with correct coordinates!)


from pdbbind_dock import PDBresult
from tqdm import tqdm
import zarr
import pandas as pd
from rdkit.Chem import Descriptors
import math
import os
import json
from plinder.core.scores import query_index

store = zarr.open('formatted_results.zarr', 'r')

lppdbbind_csv = pd.read_csv('LP_PDBBind.csv', index_col=0)

with open('PDBbind_data_split_cleansplit.json') as f:
    clensplit = json.load(f)

with open('2016_core_set/2016_core_set.md') as file:
    for line in file:
        line = line.strip().split(',')
        hac_test_pdbs = [ pdb.strip() for pdb in line ]

with open('2016_core_set/training_set-core_test.md') as file:
    for line in file:
        line = line.strip().split(',')
        hac_train_pdbs = [ pdb.strip() for pdb in line ]

with open('2016_core_set/validation_set-core-test.md') as file:
    for line in file:
        line = line.strip().split(',')
        hac_val_pdbs = [ pdb.strip() for pdb in line ]

plinder = query_index(columns=['entry_pdb_id'], splits=["*"])
plinder = plinder.drop_duplicates('entry_pdb_id')

units = [ 'mM', 'uM', 'nM', 'pM', 'fM', 'BAD' ]
common_receptor_atoms = ["C","S","O","N","ZN","CA","CO","MG","NI","MN","FE","NA","K","SE","P","CD","H"]
common_ligand_atoms = ["C","S","O","N","CL", "P", "F", "BR", "I", "B","H"]

all_pdbs = []
for pdb in store:
    all_pdbs.append(pdb)

raw_energies = ['angle_constraint', 'atom_pair_constraint', 'chainbreak', 'coordinate_constraint', 'dihedral_constraint', 'dslf_ca_dih', 'dslf_cs_ang',
                'dslf_ss_dih', 'dslf_ss_dst', 'fa_atr', 'fa_dun', 'fa_elec', 'fa_intra_rep', 'fa_pair', 'fa_rep', 'fa_sol', 'hbond_bb_sc', 'hbond_lr_bb',
                'hbond_sc', 'hbond_sr_bb', 'omega', 'p_aa_pp', 'pro_close', 'rama', 'ref']

column_names = ['plindersplit', 'hacsplit', 'cleansplit','lppdbbindsplit','affinity_type','dG','vague_measures','covalent','fully_relaxed',
                'nres','qed','has_rare_element', 'rmsd_relax_to_crystal', 'crystal_total_score', 'relax_total_score', 'ligaway_total_score',
                'crystal_idelta_score', 'relax_idelta_score' ]

for prefix in [ 'crystal_idelta_', 'relax_idelta_' ]:
    for e in raw_energies:
        column_names.append( prefix + e )

for prefix in [ 'relax_total_', 'ligaway_total_' ]:
    for e in raw_energies:
        column_names.append( prefix + e )

dataframe = pd.DataFrame(columns=column_names)

for pdb in tqdm(all_pdbs[:]):
    pdb_entry = PDBresult( store, pdb )

    dataframe.at[pdb, 'lppdbbindsplit'] = lppdbbind_csv['new_split'][pdb]

    if pdb in clensplit['train']:
        dataframe.at[pdb, 'cleansplit'] = 'train'
    elif pdb in clensplit['casf2016_c5']:
        dataframe.at[pdb, 'cleansplit'] = 'test'
    elif pdb in clensplit['casf2013_c5']:
        dataframe.at[pdb, 'cleansplit'] = 'test'

    if pdb in hac_train_pdbs:
        dataframe.at[pdb, 'hacsplit'] = 'train'
    elif pdb in hac_test_pdbs:
        dataframe.at[pdb, 'hacsplit'] = 'test'
    elif pdb in hac_val_pdbs:
        dataframe.at[pdb, 'hacsplit'] = 'val'

    try:
        dataframe.at[pdb, 'plindersplit'] = plinder[ plinder.entry_pdb_id == pdb ][ 'split' ].values[0]
    except:
        pass
    
    for delim in [ '>=', '<=', '=', '~', '>', '<' ]:
        affinity = lppdbbind_csv['kd/ki'][pdb].split(delim)
        if len(affinity) == 2:
            break

    dataframe.at[pdb, 'vague_measures'] = delim != '='
    dataframe.at[pdb, 'affinity_type'] = affinity[0]

    val = float(affinity[1][:-2])
    unit = affinity[1][-2:]
    for u in units:
        val /= 1000
        if unit == u:
            break
    if u == 'BAD':
        print(affinity, unit)

    dataframe.at[pdb, 'dG'] = -1.987 * 298.15 * math.log( val ) / 1000

    dataframe.at[pdb, 'covalent'] = lppdbbind_csv['covalent'][pdb]

    fully_relaxed = True
    crystal = None
    relax = None
    ligaway = None
    
    if not pdb_entry.healthy:
        fully_relaxed = False
    try:
        crystal = pdb_entry.get_complex('pose')
        dataframe.at[pdb, 'nres'] = crystal.get_n_res()
    except:
        fully_relaxed = False
    try:
        relax = pdb_entry.get_complex('pose_relax')
    except:
        fully_relaxed = False
    try:
        ligaway = pdb_entry.get_complex('pose_relax_ligaway')
    except:
        fully_relaxed = False

    dataframe.at[pdb, 'fully_relaxed'] = fully_relaxed

    try:
        mol = crystal.rdkit_mol
        dataframe.at[pdb, 'qed'] = Descriptors.qed(mol)
    except:
        pass
        
    
    has_rare_element = False
    try:
        pdb_string = crystal.pdb
        receptor_elems = set()
        ligand_elems = set()
        for line in pdb_string.split('\n'):
            if line[:4] == 'ATOM':
                receptor_elems.add( line[76:78].upper().strip() )
            elif line[:6] == 'HETATM' and line[17:20] == 'UNK':
                ligand_elems.add( line[76:78].upper().strip() )
        for e in receptor_elems:
            if e not in common_receptor_atoms:
                has_rare_element = True
        for e in ligand_elems:
            if e not in common_ligand_atoms:
                has_rare_element = True
        dataframe.at[pdb, 'has_rare_element'] = has_rare_element
    except:
        pass

    try:
        dataframe.at[pdb, 'crystal_total_score'] = crystal.total_score
        dataframe.at[pdb, 'crystal_idelta_score'] = crystal.idelta_score
        for e in raw_energies:
            dataframe.at[pdb, 'crystal_idelta_'+e] = crystal.raw_delta_energies[e]
    except:
        pass

    try:
        dataframe.at[pdb, 'rmsd_relax_to_crystal'] = relax.rmsd_to_crystal
        dataframe.at[pdb, 'relax_total_score'] = relax.total_score
        dataframe.at[pdb, 'relax_idelta_score'] = relax.idelta_score
        for e in raw_energies:
            dataframe.at[pdb, 'relax_idelta_'+e] = relax.raw_delta_energies[e]
            dataframe.at[pdb, 'relax_total_'+e] = relax.raw_energies[e]
    except:
        pass

    try:
        dataframe.at[pdb, 'ligaway_total_score'] = ligaway.total_score
        for e in raw_energies:
            dataframe.at[pdb, 'ligaway_total_'+e] = ligaway.raw_energies[e]
    except:
        pass

    path = 'pdbbind_rosetta/' + pdb + '/'

    try:
        os.mkdir(path)
    except FileExistsError:
        pass

    try:
        crystal.write_pdb(path + pdb + '_crystal.pdb')
        relax.write_pdb(path + pdb + '_relax.pdb')
        ligaway.write_pdb(path + pdb + '_ligaway.pdb')
    except:
        pass

    try:
        crystal.write_mol(path + pdb + '_crystal.sdf')
        relax.write_mol(path + pdb + '_relax.sdf')
    except:
        pass

dataframe.to_csv( 'pdbbind_rosetta.csv' )