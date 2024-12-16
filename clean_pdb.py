#!/usr/bin/env python
''' This script cleans PDBs for Rosetta by removing extraneous information, converting residues names and renumbering.

It outputs both the cleaned PDB and a fasta file of the cleaned sequence.

Required parameters are the name of the PDB you want to clean, and the chain ids of the chains you want.

The PDB name may be specified with or without the .pdb file handle and may be provided as a gziped file.
If the PDB isn't found locally, the given 4 letter code will be fetched from the internet.

Chain id: only the specified chains will be extracted. You may specify more than one: "AB" gets you chain A and B,
and "C" gets you just chain C. Special notations are "nochain" to remove chain identiry from the output, and "ignorechain"
to get all the chains.

(Script written by Phil Bradley, Rhiju Das, Michael Tyka, TJ Brunette, and James Thompson from the Baker Lab. Edits done by Steven Combs, Sam Deluca, Jordan Willis and Rocco Moretti from the Meiler Lab.)
'''

# Function of this script: "clean" raw pdb file by following tasks so that rosetta modeling becomes easier

## starts residue number at 1
## translates certain residues to their cannonical amino acid equivalents
## removes unknown residues
## removes residues with 0 occupancy
## generates a fasta file
## and leaves the 1st model among many NMR models

from __future__ import print_function
import sys
import os
from sys import argv, stderr, stdout
from os import popen, system
from os.path import exists, basename
from optparse import OptionParser

# amino acid info
longer_names={'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
              'CYS': 'C', 'GLU': 'E', 'GLN': 'Q', 'GLY': 'G',
              'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
              'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
              'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'}

modres={
'0CS':'ALA',                    ##  0CS ALA  3-[(S)-HYDROPEROXYSULFINYL]-L-ALANINE
'1AB':'PRO',                    ##  1AB PRO  1,4-DIDEOXY-1,4-IMINO-D-ARABINITOL
'1LU':'LEU',                    ##  1LU LEU  4-METHYL-PENTANOIC ACID-2-OXYL GROUP
'1PA':'PHE',                    ##  1PA PHE  PHENYLMETHYLACETIC ACID ALANINE
'1TQ':'TRP',                    ##  1TQ TRP  6-(FORMYLAMINO)-7-HYDROXY-L-TRYPTOPHAN
'1TY':'TYR',                    ##  1TY TYR
'23F':'PHE',                    ##  23F PHE  (2Z)-2-AMINO-3-PHENYLACRYLIC ACID
'23S':'TRP',                    ##  23S TRP  MODIFIED TRYPTOPHAN
'2BU':'ALA',                    ##  2BU ADE
'2ML':'LEU',                    ##  2ML LEU  2-METHYLLEUCINE
'2MR':'ARG',                    ##  2MR ARG  N3, N4-DIMETHYLARGININE
'2MT':'PRO',                    ##  2MT PRO
'2OP':'ALA',                    ##  2OP (2S  2-HYDROXYPROPANAL
'2TY':'TYR',                    ##  2TY TYR
'32S':'TRP',                    ##  32S TRP  MODIFIED TRYPTOPHAN
'32T':'TRP',                    ##  32T TRP  MODIFIED TRYPTOPHAN
'3AH':'HIS',                    ##  3AH HIS
'3MD':'ASP',                    ##  3MD ASP  2S,3S-3-METHYLASPARTIC ACID
'3TY':'TYR',                    ##  3TY TYR  MODIFIED TYROSINE
'4DP':'TRP',                    ##  4DP TRP
'4F3':'ALA',                    ##  4F3 ALA  CYCLIZED
'4FB':'PRO',                    ##  4FB PRO  (4S)-4-FLUORO-L-PROLINE
'4FW':'TRP',                    ##  4FW TRP  4-FLUOROTRYPTOPHANE
'4HT':'TRP',                    ##  4HT TRP  4-HYDROXYTRYPTOPHAN
'4IN':'TRP',                    ##  4IN TRP  4-AMINO-L-TRYPTOPHAN
'4PH':'PHE',                    ##  4PH PHE  4-METHYL-L-PHENYLALANINE
'5CS':'CYS',                    ##  5CS CYS
'6CL':'LYS',                    ##  6CL LYS  6-CARBOXYLYSINE
'6CW':'TRP',                    ##  6CW TRP  6-CHLORO-L-TRYPTOPHAN
'A0A':'ASP',                    ##  A0A ASP  ASPARTYL-FORMYL MIXED ANHYDRIDE
'AA4':'ALA',                    ##  AA4 ALA  2-AMINO-5-HYDROXYPENTANOIC ACID
'AAR':'ARG',                    ##  AAR ARG  ARGININEAMIDE
'AB7':'GLU',                    ##  AB7 GLU  ALPHA-AMINOBUTYRIC ACID
'ABA':'ALA',                    ##  ABA ALA  ALPHA-AMINOBUTYRIC ACID
'ACB':'ASP',                    ##  ACB ASP  3-METHYL-ASPARTIC ACID
'ACL':'ARG',                    ##  ACL ARG  DEOXY-CHLOROMETHYL-ARGININE
'ACY':'GLY',                    ##  ACY GLY  POST-TRANSLATIONAL MODIFICATION
'AEI':'THR',                    ##  AEI THR  ACYLATED THR
'AFA':'ASN',                    ##  AFA ASN  N-[7-METHYL-OCT-2,4-DIENOYL]ASPARAGINE
'AGM':'ARG',                    ##  AGM ARG  4-METHYL-ARGININE
'AGT':'CYS',                    ##  AGT CYS  AGMATINE-CYSTEINE ADDUCT
'AHB':'ASN',                    ##  AHB ASN  BETA-HYDROXYASPARAGINE
'AHO':'ALA',                    ##  AHO ALA  N-ACETYL-N-HYDROXY-L-ORNITHINE
'AHP':'ALA',                    ##  AHP ALA  2-AMINO-HEPTANOIC ACID
'AIB':'ALA',                    ##  AIB ALA  ALPHA-AMINOISOBUTYRIC ACID
'AKL':'ASP',                    ##  AKL ASP  3-AMINO-5-CHLORO-4-OXOPENTANOIC ACID
'ALA':'ALA',                    ##  ALA ALA
'ALC':'ALA',                    ##  ALC ALA  2-AMINO-3-CYCLOHEXYL-PROPIONIC ACID
'ALG':'ARG',                    ##  ALG ARG  GUANIDINOBUTYRYL GROUP
'ALM':'ALA',                    ##  ALM ALA  1-METHYL-ALANINAL
'ALN':'ALA',                    ##  ALN ALA  NAPHTHALEN-2-YL-3-ALANINE
'ALO':'THR',                    ##  ALO THR  ALLO-THREONINE
'ALS':'ALA',                    ##  ALS ALA  2-AMINO-3-OXO-4-SULFO-BUTYRIC ACID
'ALT':'ALA',                    ##  ALT ALA  THIOALANINE
'ALY':'LYS',                    ##  ALY LYS  N(6)-ACETYLLYSINE
'AME':'MET',                    ##  AME MET  ACETYLATED METHIONINE
'AP7':'ALA',                    ##  AP7 ADE
'APH':'ALA',                    ##  APH ALA  P-AMIDINOPHENYL-3-ALANINE
'API':'LYS',                    ##  API LYS  2,6-DIAMINOPIMELIC ACID
'APK':'LYS',                    ##  APK LYS
'AR2':'ARG',                    ##  AR2 ARG  ARGINYL-BENZOTHIAZOLE-6-CARBOXYLIC ACID
'AR4':'GLU',                    ##  AR4 GLU
'ARG':'ARG',                    ##  ARG ARG
'ARM':'ARG',                    ##  ARM ARG  DEOXY-METHYL-ARGININE
'ARO':'ARG',                    ##  ARO ARG  C-GAMMA-HYDROXY ARGININE
'ASA':'ASP',                    ##  ASA ASP  ASPARTIC ALDEHYDE
'ASB':'ASP',                    ##  ASB ASP  ASPARTIC ACID-4-CARBOXYETHYL ESTER
'ASI':'ASP',                    ##  ASI ASP  L-ISO-ASPARTATE
'ASK':'ASP',                    ##  ASK ASP  DEHYDROXYMETHYLASPARTIC ACID
'ASL':'ASP',                    ##  ASL ASP  ASPARTIC ACID-4-CARBOXYETHYL ESTER
'ASN':'ASN',                    ##  ASN ASN
'ASP':'ASP',                    ##  ASP ASP
'AYA':'ALA',                    ##  AYA ALA  N-ACETYLALANINE
'AYG':'ALA',                    ##  AYG ALA
'AZK':'LYS',                    ##  AZK LYS  (2S)-2-AMINO-6-TRIAZANYLHEXAN-1-OL
'B2A':'ALA',                    ##  B2A ALA  ALANINE BORONIC ACID
'B2F':'PHE',                    ##  B2F PHE  PHENYLALANINE BORONIC ACID
'B2I':'ILE',                    ##  B2I ILE  ISOLEUCINE BORONIC ACID
'B2V':'VAL',                    ##  B2V VAL  VALINE BORONIC ACID
'B3A':'ALA',                    ##  B3A ALA  (3S)-3-AMINOBUTANOIC ACID
'B3D':'ASP',                    ##  B3D ASP  3-AMINOPENTANEDIOIC ACID
'B3E':'GLU',                    ##  B3E GLU  (3S)-3-AMINOHEXANEDIOIC ACID
'B3K':'LYS',                    ##  B3K LYS  (3S)-3,7-DIAMINOHEPTANOIC ACID
'B3S':'SER',                    ##  B3S SER  (3R)-3-AMINO-4-HYDROXYBUTANOIC ACID
'B3X':'ASN',                    ##  B3X ASN  (3S)-3,5-DIAMINO-5-OXOPENTANOIC ACID
'B3Y':'TYR',                    ##  B3Y TYR
'BAL':'ALA',                    ##  BAL ALA  BETA-ALANINE
'BBC':'CYS',                    ##  BBC CYS
'BCS':'CYS',                    ##  BCS CYS  BENZYLCYSTEINE
'BCX':'CYS',                    ##  BCX CYS  BETA-3-CYSTEINE
'BFD':'ASP',                    ##  BFD ASP  ASPARTATE BERYLLIUM FLUORIDE
'BG1':'SER',                    ##  BG1 SER
'BHD':'ASP',                    ##  BHD ASP  BETA-HYDROXYASPARTIC ACID
'BIF':'PHE',                    ##  BIF PHE
'BLE':'LEU',                    ##  BLE LEU  LEUCINE BORONIC ACID
'BLY':'LYS',                    ##  BLY LYS  LYSINE BORONIC ACID
'BMT':'THR',                    ##  BMT THR
'BNN':'ALA',                    ##  BNN ALA  ACETYL-P-AMIDINOPHENYLALANINE
'BOR':'ARG',                    ##  BOR ARG
'BPE':'CYS',                    ##  BPE CYS
'BTR':'TRP',                    ##  BTR TRP  6-BROMO-TRYPTOPHAN
'BUC':'CYS',                    ##  BUC CYS  S,S-BUTYLTHIOCYSTEINE
'BUG':'LEU',                    ##  BUG LEU  TERT-LEUCYL AMINE
'C12':'ALA',                    ##  C12 ALA
'C1X':'LYS',                    ##  C1X LYS  MODIFIED LYSINE
'C3Y':'CYS',                    ##  C3Y CYS  MODIFIED CYSTEINE
'C5C':'CYS',                    ##  C5C CYS  S-CYCLOPENTYL THIOCYSTEINE
'C6C':'CYS',                    ##  C6C CYS  S-CYCLOHEXYL THIOCYSTEINE
'C99':'ALA',                    ##  C99 ALA
'CAB':'ALA',                    ##  CAB ALA  4-CARBOXY-4-AMINOBUTANAL
'CAF':'CYS',                    ##  CAF CYS  S-DIMETHYLARSINOYL-CYSTEINE
'CAS':'CYS',                    ##  CAS CYS  S-(DIMETHYLARSENIC)CYSTEINE
'CCS':'CYS',                    ##  CCS CYS  CARBOXYMETHYLATED CYSTEINE
'CGU':'GLU',                    ##  CGU GLU  CARBOXYLATION OF THE CG ATOM
'CH6':'ALA',                    ##  CH6 ALA
'CH7':'ALA',                    ##  CH7 ALA
'CHG':'GLY',                    ##  CHG GLY  CYCLOHEXYL GLYCINE
'CHP':'GLY',                    ##  CHP GLY  3-CHLORO-4-HYDROXYPHENYLGLYCINE
'CHS':'PHE',                    ##  CHS PHE  4-AMINO-5-CYCLOHEXYL-3-HYDROXY-PENTANOIC AC
'CIR':'ARG',                    ##  CIR ARG  CITRULLINE
'CLB':'ALA',                    ##  CLB ALA
'CLD':'ALA',                    ##  CLD ALA
'CLE':'LEU',                    ##  CLE LEU  LEUCINE AMIDE
'CLG':'LYS',                    ##  CLG LYS
'CLH':'LYS',                    ##  CLH LYS
'CLV':'ALA',                    ##  CLV ALA
'CME':'CYS',                    ##  CME CYS  MODIFIED CYSTEINE
'CML':'CYS',                    ##  CML CYS
'CMT':'CYS',                    ##  CMT CYS  O-METHYLCYSTEINE
'CQR':'ALA',                    ##  CQR ALA
'CR2':'ALA',                    ##  CR2 ALA  POST-TRANSLATIONAL MODIFICATION
'CR5':'ALA',                    ##  CR5 ALA
'CR7':'ALA',                    ##  CR7 ALA
'CR8':'ALA',                    ##  CR8 ALA
'CRK':'ALA',                    ##  CRK ALA
'CRO':'THR',                    ##  CRO THR  CYCLIZED
'CRQ':'TYR',                    ##  CRQ TYR
'CRW':'ALA',                    ##  CRW ALA
'CRX':'ALA',                    ##  CRX ALA
'CS1':'CYS',                    ##  CS1 CYS  S-(2-ANILINYL-SULFANYL)-CYSTEINE
'CS3':'CYS',                    ##  CS3 CYS
'CS4':'CYS',                    ##  CS4 CYS
'CSA':'CYS',                    ##  CSA CYS  S-ACETONYLCYSTEIN
'CSB':'CYS',                    ##  CSB CYS  CYS BOUND TO LEAD ION
'CSD':'CYS',                    ##  CSD CYS  3-SULFINOALANINE
'CSE':'CYS',                    ##  CSE CYS  SELENOCYSTEINE
'CSI':'ALA',                    ##  CSI ALA
'CSO':'CYS',                    ##  CSO CYS  INE S-HYDROXYCYSTEINE
'CSR':'CYS',                    ##  CSR CYS  S-ARSONOCYSTEINE
'CSS':'CYS',                    ##  CSS CYS  1,3-THIAZOLE-4-CARBOXYLIC ACID
'CSU':'CYS',                    ##  CSU CYS  CYSTEINE-S-SULFONIC ACID
'CSW':'CYS',                    ##  CSW CYS  CYSTEINE-S-DIOXIDE
'CSX':'CYS',                    ##  CSX CYS  OXOCYSTEINE
'CSY':'ALA',                    ##  CSY ALA  MODIFIED TYROSINE COMPLEX
'CSZ':'CYS',                    ##  CSZ CYS  S-SELANYL CYSTEINE
'CTH':'THR',                    ##  CTH THR  4-CHLOROTHREONINE
'CWR':'ALA',                    ##  CWR ALA
'CXM':'MET',                    ##  CXM MET  N-CARBOXYMETHIONINE
'CY0':'CYS',                    ##  CY0 CYS  MODIFIED CYSTEINE
'CY1':'CYS',                    ##  CY1 CYS  ACETAMIDOMETHYLCYSTEINE
'CY3':'CYS',                    ##  CY3 CYS  2-AMINO-3-MERCAPTO-PROPIONAMIDE
'CY4':'CYS',                    ##  CY4 CYS  S-BUTYRYL-CYSTEIN
'CY7':'CYS',                    ##  CY7 CYS  MODIFIED CYSTEINE
'CYD':'CYS',                    ##  CYD CYS
'CYF':'CYS',                    ##  CYF CYS  FLUORESCEIN LABELLED CYS380 (P14)
'CYG':'CYS',                    ##  CYG CYS
'CYJ':'LYS',                    ##  CYJ LYS  MODIFIED LYSINE
'CYQ':'CYS',                    ##  CYQ CYS
'CYR':'CYS',                    ##  CYR CYS
'CYS':'CYS',                    ##  CYS CYS
'CYX':'CYS',                    ##  CYX CYS  Not an official code, but used by some programs as disulfide residues
'CZ2':'CYS',                    ##  CZ2 CYS  S-(DIHYDROXYARSINO)CYSTEINE
'CZZ':'CYS',                    ##  CZZ CYS  THIARSAHYDROXY-CYSTEINE
'DA2':'ARG',                    ##  DA2 ARG  MODIFIED ARGININE
'DAB':'ALA',                    ##  DAB ALA  2,4-DIAMINOBUTYRIC ACID
'DAH':'PHE',                    ##  DAH PHE  3,4-DIHYDROXYDAHNYLALANINE
'DAL':'ALA',                    ##  DAL ALA  D-ALANINE
'DAM':'ALA',                    ##  DAM ALA  N-METHYL-ALPHA-BETA-DEHYDROALANINE
'DAR':'ARG',                    ##  DAR ARG  D-ARGININE
'DAS':'ASP',                    ##  DAS ASP  D-ASPARTIC ACID
'DBU':'ALA',                    ##  DBU ALA  (2E)-2-AMINOBUT-2-ENOIC ACID
'DBY':'TYR',                    ##  DBY TYR  3,5 DIBROMOTYROSINE
'DBZ':'ALA',                    ##  DBZ ALA  3-(BENZOYLAMINO)-L-ALANINE
'DCL':'LEU',                    ##  DCL LEU  2-AMINO-4-METHYL-PENTANYL GROUP
'DCY':'CYS',                    ##  DCY CYS  D-CYSTEINE
'DDE':'HIS',                    ##  DDE HIS
'DGL':'GLU',                    ##  DGL GLU  D-GLU
'DGN':'GLN',                    ##  DGN GLN  D-GLUTAMINE
'DHA':'ALA',                    ##  DHA ALA  2-AMINO-ACRYLIC ACID
'DHI':'HIS',                    ##  DHI HIS  D-HISTIDINE
'DHL':'SER',                    ##  DHL SER  POST-TRANSLATIONAL MODIFICATION
'DIL':'ILE',                    ##  DIL ILE  D-ISOLEUCINE
'DIV':'VAL',                    ##  DIV VAL  D-ISOVALINE
'DLE':'LEU',                    ##  DLE LEU  D-LEUCINE
'DLS':'LYS',                    ##  DLS LYS  DI-ACETYL-LYSINE
'DLY':'LYS',                    ##  DLY LYS  D-LYSINE
'DMH':'ASN',                    ##  DMH ASN  N4,N4-DIMETHYL-ASPARAGINE
'DMK':'ASP',                    ##  DMK ASP  DIMETHYL ASPARTIC ACID
'DNE':'LEU',                    ##  DNE LEU  D-NORLEUCINE
'DNG':'LEU',                    ##  DNG LEU  N-FORMYL-D-NORLEUCINE
'DNL':'LYS',                    ##  DNL LYS  6-AMINO-HEXANAL
'DNM':'LEU',                    ##  DNM LEU  D-N-METHYL NORLEUCINE
'DPH':'PHE',                    ##  DPH PHE  DEAMINO-METHYL-PHENYLALANINE
'DPL':'PRO',                    ##  DPL PRO  4-OXOPROLINE
'DPN':'PHE',                    ##  DPN PHE  D-CONFIGURATION
'DPP':'ALA',                    ##  DPP ALA  DIAMMINOPROPANOIC ACID
'DPQ':'TYR',                    ##  DPQ TYR  TYROSINE DERIVATIVE
'DPR':'PRO',                    ##  DPR PRO  D-PROLINE
'DSE':'SER',                    ##  DSE SER  D-SERINE N-METHYLATED
'DSG':'ASN',                    ##  DSG ASN  D-ASPARAGINE
'DSN':'SER',                    ##  DSN SER  D-SERINE
'DTH':'THR',                    ##  DTH THR  D-THREONINE
'DTR':'TRP',                    ##  DTR TRP  D-TRYPTOPHAN
'DTY':'TYR',                    ##  DTY TYR  D-TYROSINE
'DVA':'VAL',                    ##  DVA VAL  D-VALINE
'DYG':'ALA',                    ##  DYG ALA
'DYS':'CYS',                    ##  DYS CYS
'EFC':'CYS',                    ##  EFC CYS  S,S-(2-FLUOROETHYL)THIOCYSTEINE
'ESB':'TYR',                    ##  ESB TYR
'ESC':'MET',                    ##  ESC MET  2-AMINO-4-ETHYL SULFANYL BUTYRIC ACID
'FCL':'PHE',                    ##  FCL PHE  3-CHLORO-L-PHENYLALANINE
'FGL':'ALA',                    ##  FGL ALA  2-AMINOPROPANEDIOIC ACID
'FGP':'SER',                    ##  FGP SER
'FHL':'LYS',                    ##  FHL LYS  MODIFIED LYSINE
'FLE':'LEU',                    ##  FLE LEU  FUROYL-LEUCINE
'FLT':'TYR',                    ##  FLT TYR  FLUOROMALONYL TYROSINE
'FME':'MET',                    ##  FME MET  FORMYL-METHIONINE
'FOE':'CYS',                    ##  FOE CYS
'FOG':'PHE',                    ##  FOG PHE  PHENYLALANINOYL-[1-HYDROXY]-2-PROPYLENE
'FOR':'MET',                    ##  FOR MET
'FRF':'PHE',                    ##  FRF PHE  PHE FOLLOWED BY REDUCED PHE
'FTR':'TRP',                    ##  FTR TRP  FLUOROTRYPTOPHANE
'FTY':'TYR',                    ##  FTY TYR  DEOXY-DIFLUOROMETHELENE-PHOSPHOTYROSINE
'GHG':'GLN',                    ##  GHG GLN  GAMMA-HYDROXY-GLUTAMINE
'GHP':'GLY',                    ##  GHP GLY  4-HYDROXYPHENYLGLYCINE
'GL3':'GLY',                    ##  GL3 GLY  POST-TRANSLATIONAL MODIFICATION
'GLH':'GLN',                    ##  GLH GLN
'GLN':'GLN',                    ##  GLN GLN
'GLU':'GLU',                    ##  GLU GLU
'GLY':'GLY',                    ##  GLY GLY
'GLZ':'GLY',                    ##  GLZ GLY  AMINO-ACETALDEHYDE
'GMA':'GLU',                    ##  GMA GLU  1-AMIDO-GLUTAMIC ACID
'GMU':'ALA',                    ##  GMU 5MU
'GPL':'LYS',                    ##  GPL LYS  LYSINE GUANOSINE-5'-MONOPHOSPHATE
'GT9':'CYS',                    ##  GT9 CYS  SG ALKYLATED
'GVL':'SER',                    ##  GVL SER  SERINE MODIFED WITH PHOSPHOPANTETHEINE
'GYC':'CYS',                    ##  GYC CYS
'GYS':'GLY',                    ##  GYS GLY
'H5M':'PRO',                    ##  H5M PRO  TRANS-3-HYDROXY-5-METHYLPROLINE
'HHK':'ALA',                    ##  HHK ALA  (2S)-2,8-DIAMINOOCTANOIC ACID
'HIA':'HIS',                    ##  HIA HIS  L-HISTIDINE AMIDE
'HIC':'HIS',                    ##  HIC HIS  4-METHYL-HISTIDINE
'HID':'HIS',                    ##  HID HIS  Not an official code, but used by some programs as delta-protonated histidine
'HIE':'HIS',                    ##  HIE HIS  Not an official code, but used by some programs as epsilon-protonated histidine
# The backbone check in clean_pdb.py will take care of "official" usage of HID and HIE residues
'HIP':'HIS',                    ##  HIP HIS  ND1-PHOSPHONOHISTIDINE, also non-officially as protonated histidine
'HIQ':'HIS',                    ##  HIQ HIS  MODIFIED HISTIDINE
'HIS':'HIS',                    ##  HIS HIS
'HLU':'LEU',                    ##  HLU LEU  BETA-HYDROXYLEUCINE
'HMF':'ALA',                    ##  HMF ALA  2-AMINO-4-PHENYL-BUTYRIC ACID
'HMR':'ARG',                    ##  HMR ARG  BETA-HOMOARGININE
'HPE':'PHE',                    ##  HPE PHE  HOMOPHENYLALANINE
'HPH':'PHE',                    ##  HPH PHE  PHENYLALANINOL GROUP
'HPQ':'PHE',                    ##  HPQ PHE  HOMOPHENYLALANINYLMETHANE
'HRG':'ARG',                    ##  HRG ARG  L-HOMOARGININE
'HSE':'SER',                    ##  HSE SER  L-HOMOSERINE
'HSL':'SER',                    ##  HSL SER  HOMOSERINE LACTONE
'HSO':'HIS',                    ##  HSO HIS  HISTIDINOL
'HTI':'CYS',                    ##  HTI CYS
'HTR':'TRP',                    ##  HTR TRP  BETA-HYDROXYTRYPTOPHANE
'HY3':'PRO',                    ##  HY3 PRO  3-HYDROXYPROLINE
'HYP':'PRO',                    ##  HYP PRO  4-HYDROXYPROLINE
'IAM':'ALA',                    ##  IAM ALA  4-[(ISOPROPYLAMINO)METHYL]PHENYLALANINE
'IAS':'ASP',                    ##  IAS ASP  ASPARTYL GROUP
'IGL':'ALA',                    ##  IGL ALA  ALPHA-AMINO-2-INDANACETIC ACID
'IIL':'ILE',                    ##  IIL ILE  ISO-ISOLEUCINE
'ILE':'ILE',                    ##  ILE ILE
'ILG':'GLU',                    ##  ILG GLU  GLU LINKED TO NEXT RESIDUE VIA CG
'ILX':'ILE',                    ##  ILX ILE  4,5-DIHYDROXYISOLEUCINE
'IML':'ILE',                    ##  IML ILE  N-METHYLATED
'IPG':'GLY',                    ##  IPG GLY  N-ISOPROPYL GLYCINE
'IT1':'LYS',                    ##  IT1 LYS
'IYR':'TYR',                    ##  IYR TYR  3-IODO-TYROSINE
'KCX':'LYS',                    ##  KCX LYS  CARBAMOYLATED LYSINE
'KGC':'LYS',                    ##  KGC LYS
'KOR':'CYS',                    ##  KOR CYS  MODIFIED CYSTEINE
'KST':'LYS',                    ##  KST LYS  N~6~-(5-CARBOXY-3-THIENYL)-L-LYSINE
'KYN':'ALA',                    ##  KYN ALA  KYNURENINE
'LA2':'LYS',                    ##  LA2 LYS
'LAL':'ALA',                    ##  LAL ALA  N,N-DIMETHYL-L-ALANINE
'LCK':'LYS',                    ##  LCK LYS
'LCX':'LYS',                    ##  LCX LYS  CARBAMYLATED LYSINE
'LDH':'LYS',                    ##  LDH LYS  N~6~-ETHYL-L-LYSINE
'LED':'LEU',                    ##  LED LEU  POST-TRANSLATIONAL MODIFICATION
'LEF':'LEU',                    ##  LEF LEU  2-5-FLUOROLEUCINE
'LET':'LYS',                    ##  LET LYS  ODIFIED LYSINE
'LEU':'LEU',                    ##  LEU LEU
'LLP':'LYS',                    ##  LLP LYS
'LLY':'LYS',                    ##  LLY LYS  NZ-(DICARBOXYMETHYL)LYSINE
'LME':'GLU',                    ##  LME GLU  (3R)-3-METHYL-L-GLUTAMIC ACID
'LNT':'LEU',                    ##  LNT LEU
'LPD':'PRO',                    ##  LPD PRO  L-PROLINAMIDE
'LSO':'LYS',                    ##  LSO LYS  MODIFIED LYSINE
'LYM':'LYS',                    ##  LYM LYS  DEOXY-METHYL-LYSINE
'LYN':'LYS',                    ##  LYN LYS  2,6-DIAMINO-HEXANOIC ACID AMIDE
'LYP':'LYS',                    ##  LYP LYS  N~6~-METHYL-N~6~-PROPYL-L-LYSINE
'LYR':'LYS',                    ##  LYR LYS  MODIFIED LYSINE
'LYS':'LYS',                    ##  LYS LYS
'LYX':'LYS',                    ##  LYX LYS  N''-(2-COENZYME A)-PROPANOYL-LYSINE
'LYZ':'LYS',                    ##  LYZ LYS  5-HYDROXYLYSINE
'M0H':'CYS',                    ##  M0H CYS  S-(HYDROXYMETHYL)-L-CYSTEINE
'M2L':'LYS',                    ##  M2L LYS
'M3L':'LYS',                    ##  M3L LYS  N-TRIMETHYLLYSINE
'MAA':'ALA',                    ##  MAA ALA  N-METHYLALANINE
'MAI':'ARG',                    ##  MAI ARG  DEOXO-METHYLARGININE
'MBQ':'TYR',                    ##  MBQ TYR
'MC1':'SER',                    ##  MC1 SER  METHICILLIN ACYL-SERINE
'MCL':'LYS',                    ##  MCL LYS  NZ-(1-CARBOXYETHYL)-LYSINE
'MCS':'CYS',                    ##  MCS CYS  MALONYLCYSTEINE
'MDO':'ALA',                    ##  MDO ALA
'MEA':'PHE',                    ##  MEA PHE  N-METHYLPHENYLALANINE
'MEG':'GLU',                    ##  MEG GLU  (2S,3R)-3-METHYL-GLUTAMIC ACID
'MEN':'ASN',                    ##  MEN ASN  GAMMA METHYL ASPARAGINE
'MET':'MET',                    ##  MET MET
'MEU':'GLY',                    ##  MEU GLY  O-METHYL-GLYCINE
'MFC':'ALA',                    ##  MFC ALA  CYCLIZED
'MGG':'ARG',                    ##  MGG ARG  MODIFIED D-ARGININE
'MGN':'GLN',                    ##  MGN GLN  2-METHYL-GLUTAMINE
'MHL':'LEU',                    ##  MHL LEU  N-METHYLATED, HYDROXY
'MHO':'MET',                    ##  MHO MET  POST-TRANSLATIONAL MODIFICATION
'MHS':'HIS',                    ##  MHS HIS  1-N-METHYLHISTIDINE
'MIS':'SER',                    ##  MIS SER  MODIFIED SERINE
'MLE':'LEU',                    ##  MLE LEU  N-METHYLATED
'MLL':'LEU',                    ##  MLL LEU  METHYL L-LEUCINATE
'MLY':'LYS',                    ##  MLY LYS  METHYLATED LYSINE
'MLZ':'LYS',                    ##  MLZ LYS  N-METHYL-LYSINE
'MME':'MET',                    ##  MME MET  N-METHYL METHIONINE
'MNL':'LEU',                    ##  MNL LEU  4,N-DIMETHYLNORLEUCINE
'MNV':'VAL',                    ##  MNV VAL  N-METHYL-C-AMINO VALINE
'MPQ':'GLY',                    ##  MPQ GLY  N-METHYL-ALPHA-PHENYL-GLYCINE
'MSA':'GLY',                    ##  MSA GLY  (2-S-METHYL) SARCOSINE
'MSE':'MET',                    ##  MSE MET  ELENOMETHIONINE
'MSO':'MET',                    ##  MSO MET  METHIONINE SULFOXIDE
'MTY':'PHE',                    ##  MTY PHE  3-HYDROXYPHENYLALANINE
'MVA':'VAL',                    ##  MVA VAL  N-METHYLATED
'N10':'SER',                    ##  N10 SER  O-[(HEXYLAMINO)CARBONYL]-L-SERINE
'NAL':'ALA',                    ##  NAL ALA  BETA-(2-NAPHTHYL)-ALANINE
'NAM':'ALA',                    ##  NAM ALA  NAM NAPTHYLAMINOALANINE
'NBQ':'TYR',                    ##  NBQ TYR
'NC1':'SER',                    ##  NC1 SER  NITROCEFIN ACYL-SERINE
'NCB':'ALA',                    ##  NCB ALA  CHEMICAL MODIFICATION
'NEP':'HIS',                    ##  NEP HIS  N1-PHOSPHONOHISTIDINE
'NFA':'PHE',                    ##  NFA PHE  MODIFIED PHENYLALANINE
'NIY':'TYR',                    ##  NIY TYR  META-NITRO-TYROSINE
'NLE':'LEU',                    ##  NLE LEU  NORLEUCINE
'NLN':'LEU',                    ##  NLN LEU  NORLEUCINE AMIDE
'NLO':'LEU',                    ##  NLO LEU  O-METHYL-L-NORLEUCINE
'NMC':'GLY',                    ##  NMC GLY  N-CYCLOPROPYLMETHYL GLYCINE
'NMM':'ARG',                    ##  NMM ARG  MODIFIED ARGININE
'NPH':'CYS',                    ##  NPH CYS
'NRQ':'ALA',                    ##  NRQ ALA
'NVA':'VAL',                    ##  NVA VAL  NORVALINE
'NYC':'ALA',                    ##  NYC ALA
'NYS':'CYS',                    ##  NYS CYS
'NZH':'HIS',                    ##  NZH HIS
'OAS':'SER',                    ##  OAS SER  O-ACETYLSERINE
'OBS':'LYS',                    ##  OBS LYS  MODIFIED LYSINE
'OCS':'CYS',                    ##  OCS CYS  CYSTEINE SULFONIC ACID
'OCY':'CYS',                    ##  OCY CYS  HYDROXYETHYLCYSTEINE
'OHI':'HIS',                    ##  OHI HIS  3-(2-OXO-2H-IMIDAZOL-4-YL)-L-ALANINE
'OHS':'ASP',                    ##  OHS ASP  O-(CARBOXYSULFANYL)-4-OXO-L-HOMOSERINE
'OLT':'THR',                    ##  OLT THR  O-METHYL-L-THREONINE
'OMT':'MET',                    ##  OMT MET  METHIONINE SULFONE
'OPR':'ARG',                    ##  OPR ARG  C-(3-OXOPROPYL)ARGININE
'ORN':'ALA',                    ##  ORN ALA  ORNITHINE
'ORQ':'ARG',                    ##  ORQ ARG  N~5~-ACETYL-L-ORNITHINE
'OSE':'SER',                    ##  OSE SER  O-SULFO-L-SERINE
'OTY':'TYR',                    ##  OTY TYR
'OXX':'ASP',                    ##  OXX ASP  OXALYL-ASPARTYL ANHYDRIDE
'P1L':'CYS',                    ##  P1L CYS  S-PALMITOYL CYSTEINE
'P2Y':'PRO',                    ##  P2Y PRO  (2S)-PYRROLIDIN-2-YLMETHYLAMINE
'PAQ':'TYR',                    ##  PAQ TYR  SEE REMARK 999
'PAT':'TRP',                    ##  PAT TRP  ALPHA-PHOSPHONO-TRYPTOPHAN
'PBB':'CYS',                    ##  PBB CYS  S-(4-BROMOBENZYL)CYSTEINE
'PBF':'PHE',                    ##  PBF PHE  PARA-(BENZOYL)-PHENYLALANINE
'PCA':'PRO',                    ##  PCA PRO  5-OXOPROLINE
'PCS':'PHE',                    ##  PCS PHE  PHENYLALANYLMETHYLCHLORIDE
'PEC':'CYS',                    ##  PEC CYS  S,S-PENTYLTHIOCYSTEINE
'PF5':'PHE',                    ##  PF5 PHE  2,3,4,5,6-PENTAFLUORO-L-PHENYLALANINE
'PFF':'PHE',                    ##  PFF PHE  4-FLUORO-L-PHENYLALANINE
'PG1':'SER',                    ##  PG1 SER  BENZYLPENICILLOYL-ACYLATED SERINE
'PG9':'GLY',                    ##  PG9 GLY  D-PHENYLGLYCINE
'PHA':'PHE',                    ##  PHA PHE  PHENYLALANINAL
'PHD':'ASP',                    ##  PHD ASP  2-AMINO-4-OXO-4-PHOSPHONOOXY-BUTYRIC ACID
'PHE':'PHE',                    ##  PHE PHE
'PHI':'PHE',                    ##  PHI PHE  IODO-PHENYLALANINE
'PHL':'PHE',                    ##  PHL PHE  L-PHENYLALANINOL
'PHM':'PHE',                    ##  PHM PHE  PHENYLALANYLMETHANE
'PIA':'ALA',                    ##  PIA ALA  FUSION OF ALA 65, TYR 66, GLY 67
'PLE':'LEU',                    ##  PLE LEU  LEUCINE PHOSPHINIC ACID
'PM3':'PHE',                    ##  PM3 PHE
'POM':'PRO',                    ##  POM PRO  CIS-5-METHYL-4-OXOPROLINE
'PPH':'LEU',                    ##  PPH LEU  PHENYLALANINE PHOSPHINIC ACID
'PPN':'PHE',                    ##  PPN PHE  THE LIGAND IS A PARA-NITRO-PHENYLALANINE
'PR3':'CYS',                    ##  PR3 CYS  INE DTT-CYSTEINE
'PRO':'PRO',                    ##  PRO PRO
'PRQ':'PHE',                    ##  PRQ PHE  PHENYLALANINE
'PRR':'ALA',                    ##  PRR ALA  3-(METHYL-PYRIDINIUM)ALANINE
'PRS':'PRO',                    ##  PRS PRO  THIOPROLINE
'PSA':'PHE',                    ##  PSA PHE
'PSH':'HIS',                    ##  PSH HIS  1-THIOPHOSPHONO-L-HISTIDINE
'PTH':'TYR',                    ##  PTH TYR  METHYLENE-HYDROXY-PHOSPHOTYROSINE
'PTM':'TYR',                    ##  PTM TYR  ALPHA-METHYL-O-PHOSPHOTYROSINE
'PTR':'TYR',                    ##  PTR TYR  O-PHOSPHOTYROSINE
'PYA':'ALA',                    ##  PYA ALA  3-(1,10-PHENANTHROL-2-YL)-L-ALANINE
'PYC':'ALA',                    ##  PYC ALA  PYRROLE-2-CARBOXYLATE
'PYR':'SER',                    ##  PYR SER  CHEMICALLY MODIFIED
'PYT':'ALA',                    ##  PYT ALA  MODIFIED ALANINE
'PYX':'CYS',                    ##  PYX CYS  S-[S-THIOPYRIDOXAMINYL]CYSTEINE
'R1A':'CYS',                    ##  R1A CYS
'R1B':'CYS',                    ##  R1B CYS
'R1F':'CYS',                    ##  R1F CYS
'R7A':'CYS',                    ##  R7A CYS
'RC7':'ALA',                    ##  RC7 ALA
'RCY':'CYS',                    ##  RCY CYS
'S1H':'SER',                    ##  S1H SER  1-HEXADECANOSULFONYL-O-L-SERINE
'SAC':'SER',                    ##  SAC SER  N-ACETYL-SERINE
'SAH':'CYS',                    ##  SAH CYS  S-ADENOSYL-L-HOMOCYSTEINE
'SAR':'GLY',                    ##  SAR GLY  SARCOSINE
'SBD':'SER',                    ##  SBD SER
'SBG':'SER',                    ##  SBG SER  MODIFIED SERINE
'SBL':'SER',                    ##  SBL SER
'SC2':'CYS',                    ##  SC2 CYS  N-ACETYL-L-CYSTEINE
'SCH':'CYS',                    ##  SCH CYS  S-METHYL THIOCYSTEINE GROUP
'SCS':'CYS',                    ##  SCS CYS  MODIFIED CYSTEINE
'SCY':'CYS',                    ##  SCY CYS  CETYLATED CYSTEINE
'SDP':'SER',                    ##  SDP SER
'SEB':'SER',                    ##  SEB SER  O-BENZYLSULFONYL-SERINE
'SEC':'ALA',                    ##  SEC ALA  2-AMINO-3-SELENINO-PROPIONIC ACID
'SEL':'SER',                    ##  SEL SER  2-AMINO-1,3-PROPANEDIOL
'SEP':'SER',                    ##  SEP SER  E PHOSPHOSERINE
'SER':'SER',                    ##  SER SER
'SET':'SER',                    ##  SET SER  AMINOSERINE
'SGB':'SER',                    ##  SGB SER  MODIFIED SERINE
'SGR':'SER',                    ##  SGR SER  MODIFIED SERINE
'SHC':'CYS',                    ##  SHC CYS  S-HEXYLCYSTEINE
'SHP':'GLY',                    ##  SHP GLY  (4-HYDROXYMALTOSEPHENYL)GLYCINE
'SIC':'ALA',                    ##  SIC ALA
'SLZ':'LYS',                    ##  SLZ LYS  L-THIALYSINE
'SMC':'CYS',                    ##  SMC CYS  POST-TRANSLATIONAL MODIFICATION
'SME':'MET',                    ##  SME MET  METHIONINE SULFOXIDE
'SMF':'PHE',                    ##  SMF PHE  4-SULFOMETHYL-L-PHENYLALANINE
'SNC':'CYS',                    ##  SNC CYS  S-NITROSO CYSTEINE
'SNN':'ASP',                    ##  SNN ASP  POST-TRANSLATIONAL MODIFICATION
'SOC':'CYS',                    ##  SOC CYS  DIOXYSELENOCYSTEINE
'SOY':'SER',                    ##  SOY SER  OXACILLOYL-ACYLATED SERINE
'SUI':'ALA',                    ##  SUI ALA
'SUN':'SER',                    ##  SUN SER  TABUN CONJUGATED SERINE
'SVA':'SER',                    ##  SVA SER  SERINE VANADATE
'SVV':'SER',                    ##  SVV SER  MODIFIED SERINE
'SVX':'SER',                    ##  SVX SER  MODIFIED SERINE
'SVY':'SER',                    ##  SVY SER  MODIFIED SERINE
'SVZ':'SER',                    ##  SVZ SER  MODIFIED SERINE
'SXE':'SER',                    ##  SXE SER  MODIFIED SERINE
'TBG':'GLY',                    ##  TBG GLY  T-BUTYL GLYCINE
'TBM':'THR',                    ##  TBM THR
'TCQ':'TYR',                    ##  TCQ TYR  MODIFIED TYROSINE
'TEE':'CYS',                    ##  TEE CYS  POST-TRANSLATIONAL MODIFICATION
'TH5':'THR',                    ##  TH5 THR  O-ACETYL-L-THREONINE
'THC':'THR',                    ##  THC THR  N-METHYLCARBONYLTHREONINE
'THR':'THR',                    ##  THR THR
'TIH':'ALA',                    ##  TIH ALA  BETA(2-THIENYL)ALANINE
'TMD':'THR',                    ##  TMD THR  N-METHYLATED, EPSILON C ALKYLATED
'TNB':'CYS',                    ##  TNB CYS  S-(2,3,6-TRINITROPHENYL)CYSTEINE
'TOX':'TRP',                    ##  TOX TRP
'TPL':'TRP',                    ##  TPL TRP  TRYTOPHANOL
'TPO':'THR',                    ##  TPO THR  HOSPHOTHREONINE
'TPQ':'ALA',                    ##  TPQ ALA  2,4,5-TRIHYDROXYPHENYLALANINE
'TQQ':'TRP',                    ##  TQQ TRP
'TRF':'TRP',                    ##  TRF TRP  N1-FORMYL-TRYPTOPHAN
'TRN':'TRP',                    ##  TRN TRP  AZA-TRYPTOPHAN
'TRO':'TRP',                    ##  TRO TRP  2-HYDROXY-TRYPTOPHAN
'TRP':'TRP',                    ##  TRP TRP
'TRQ':'TRP',                    ##  TRQ TRP
'TRW':'TRP',                    ##  TRW TRP
'TRX':'TRP',                    ##  TRX TRP  6-HYDROXYTRYPTOPHAN
'TTQ':'TRP',                    ##  TTQ TRP  6-AMINO-7-HYDROXY-L-TRYPTOPHAN
'TTS':'TYR',                    ##  TTS TYR
'TY2':'TYR',                    ##  TY2 TYR  3-AMINO-L-TYROSINE
'TY3':'TYR',                    ##  TY3 TYR  3-HYDROXY-L-TYROSINE
'TYB':'TYR',                    ##  TYB TYR  TYROSINAL
'TYC':'TYR',                    ##  TYC TYR  L-TYROSINAMIDE
'TYI':'TYR',                    ##  TYI TYR  3,5-DIIODOTYROSINE
'TYN':'TYR',                    ##  TYN TYR  ADDUCT AT HYDROXY GROUP
'TYO':'TYR',                    ##  TYO TYR
'TYQ':'TYR',                    ##  TYQ TYR  AMINOQUINOL FORM OF TOPA QUINONONE
'TYR':'TYR',                    ##  TYR TYR
'TYS':'TYR',                    ##  TYS TYR  INE SULPHONATED TYROSINE
'TYT':'TYR',                    ##  TYT TYR
'TYX':'CYS',                    ##  TYX CYS  S-(2-ANILINO-2-OXOETHYL)-L-CYSTEINE
'TYY':'TYR',                    ##  TYY TYR  IMINOQUINONE FORM OF TOPA QUINONONE
'TYZ':'ARG',                    ##  TYZ ARG  PARA ACETAMIDO BENZOIC ACID
'UMA':'ALA',                    ##  UMA ALA
'VAD':'VAL',                    ##  VAD VAL  DEAMINOHYDROXYVALINE
'VAF':'VAL',                    ##  VAF VAL  METHYLVALINE
'VAL':'VAL',                    ##  VAL VAL
'VDL':'VAL',                    ##  VDL VAL  (2R,3R)-2,3-DIAMINOBUTANOIC ACID
'VLL':'VAL',                    ##  VLL VAL  (2S)-2,3-DIAMINOBUTANOIC ACID
'VME':'VAL',                    ##  VME VAL  O- METHYLVALINE
'X9Q':'ALA',                    ##  X9Q ALA
'XX1':'LYS',                    ##  XX1 LYS  N~6~-7H-PURIN-6-YL-L-LYSINE
'XXY':'ALA',                    ##  XXY ALA
'XYG':'ALA',                    ##  XYG ALA
'YCM':'CYS',                    ##  YCM CYS  S-(2-AMINO-2-OXOETHYL)-L-CYSTEINE
'YOF':'TYR' 
}                   ##  YOF TYR  3-FLUOROTYROSINE

class Cleaner:

    def __init__(self):
        # remote host for downloading pdbs
        self.remote_host = ''

        self.shit_stat_insres = False
        self.shit_stat_altpos = False
        self.shit_stat_modres = False
        self.shit_stat_misdns = False  # missing density!

        self.fastaseq = {}
        self.pdbfile = ""

        self.files_to_unlink = []



    def download_pdb(self, pdb_id, dest_dir):
        # print("downloading %s" % ( pdb_id ))
        url = 'http://www.rcsb.org/pdb/files/%s.pdb.gz' % (pdb_id.upper())
        dest = '%s/%s.pdb.gz' % (os.path.abspath(dest_dir), pdb_id)
        wget_cmd = 'wget --quiet %s -O %s' % (url, dest)
        print( wget_cmd )
        if self.remote_host:
            wget_cmd = 'ssh %s %s' % (self.remote_host, wget_cmd)

        lines = popen(wget_cmd).readlines()
        if (exists(dest)):
            return(dest)
        else:
            print( "Error: didn't download file!" )


    def check_and_print_pdb(self, count, residue_buffer, residue_letter):
        self.pdbfile
        # Check that CA, N and C are present!def check_and_print_pdb( outid, residue_buffer )
        hasCA = False
        hasN = False
        hasC = False
        for line in residue_buffer:
            atomname = line[12:16]
            # Only add bb atoms if they have occupancy!
            occupancy = float(line[55:60])
            if atomname == " CA " and occupancy > 0.0:
                hasCA = True
            if atomname == " N  " and occupancy > 0.0:
                hasN = True
            if atomname == " C  " and occupancy > 0.0:
                hasC = True

        # if all three backbone atoms are present withoccupancy proceed to print the residue
        if hasCA and hasN and hasC:
            for line in residue_buffer:
                # add linear residue count
                newnum = '%4d ' % count
                line_edit = line[0:22] + newnum + line[27:]
                # write the residue line
                self.pdbfile = self.pdbfile + line_edit

        # finally print residue letter into fasta strea
            chain = line[21]
            try:
                self.fastaseq[chain] += residue_letter
            except KeyError:
                self.fastaseq[chain] = residue_letter
        # count up residue number
            count = count + 1
            return True
        return False

    def get_pdb_filename( self, name ):
        '''Tries various things to get the filename to use.
        Returns None if no acceptable file exists.'''
        if( os.path.exists( name ) ):
            return name
        if( os.path.exists( name + '.pdb' ) ):
            return name + '.pdb'
        if( os.path.exists( name + '.pdb.gz' ) ):
            return name + '.pdb.gz'
        if( os.path.exists( name + '.pdb1.gz' ) ):
            return name + '.pdb1.gz'
        name = name.upper()
        if( os.path.exists( name ) ):
            return name
        if( os.path.exists( name + '.pdb' ) ):
            return name + '.pdb'
        if( os.path.exists( name + '.pdb.gz' ) ):
            return name + '.pdb.gz'
        if( os.path.exists( name + '.pdb1.gz' ) ):
            return name + '.pdb1.gz'
        # No acceptable file found
        return None


    def open_pdb( self, name ):
        '''Open the PDB given in the filename (or equivalent).
        If the file is not found, then try downloading it from the internet.

        Returns: (lines, filename_stem)
        '''
        filename = self.get_pdb_filename( name )
        if filename is None:
            print( "File for %s doesn't exist, downloading from internet." % (name) )
            filename = self.download_pdb(name[0:4].upper(), '.')
            self.files_to_unlink.append(filename)

        stem = os.path.basename(filename)
        if stem[-3:] == '.gz':
            stem = stem[:-3]
        if stem[-5:] == '.pdb1':
            stem = stem[:-5]
        if stem[-4:] == '.pdb':
            stem = stem[:-4]

        if filename[-3:] == '.gz':
            lines = popen('zcat '+filename, 'r').readlines()
        else:
            lines = open(filename, 'r').readlines()

        return lines, stem

    def apply(self, filename, chains=""):

        # set options
        allchains = False
        if chains == "":
            allchains = True

        # reset cleaner
        self.remote_host = ''

        self.shit_stat_insres = False
        self.shit_stat_altpos = False
        self.shit_stat_modres = False
        self.shit_stat_misdns = False  # missing density!

        self.fastaseq = {}
        self.pdbfile = ""

        self.files_to_unlink = []


        # start cleaning
        chainid = chains.upper()

        lines, filename_stem = self.open_pdb( filename )

        oldresnum = '   '
        count = 1

        residue_buffer = []
        residue_letter = ''

        if chainid == '_':
            chainid = ' '

        for line in lines:

            if line.startswith('ENDMDL'): break  # Only take the first NMR model
            if len(line) > 21 and ( line[21] in chainid or allchains):
                if line[0:4] != "ATOM" and line[0:6] != 'HETATM':
                    continue

                line_edit = line
                resn = line[17:20]

                # Is it a modified residue ?
                # (Looking for modified residues in both ATOM and HETATM records is deliberate)
                if resn in modres:
                    # if so replace it with its canonical equivalent !
                    orig_resn = resn
                    resn = modres[resn]
                    line_edit = 'ATOM  '+line[6:17]+ resn + line[20:]

                    if orig_resn == "MSE":
                        # don't count MSE as modified residues for flagging purposes (because they're so common)
                        # Also, fix up the selenium atom naming
                        if (line_edit[12:14] == 'SE'):
                            line_edit = line_edit[0:12]+' S'+line_edit[14:]
                        if len(line_edit) > 75:
                            if (line_edit[76:78] == 'SE'):
                                line_edit = line_edit[0:76]+' S'+line_edit[78:]
                    else:
                        self.shit_stat_modres = True

                # Only process residues we know are valid.
                if resn not in longer_names:
                    continue

                resnum = line_edit[22:27]

                # Is this a new residue
                if not resnum == oldresnum:
                    if residue_buffer != []:  # is there a residue in the buffer ?
                        if not self.check_and_print_pdb(count, residue_buffer, residue_letter):
                            # if unsuccessful
                            self.shit_stat_misdns = True
                        else:
                            count = count + 1

                    residue_buffer = []
                    residue_letter = longer_names[resn]

                oldresnum = resnum

                insres = line[26]
                if insres != ' ':
                    self.shit_stat_insres = True

                altpos = line[16]
                if altpos != ' ':
                    self.shit_stat_altpos = True
                    if altpos == 'A':
                        line_edit = line_edit[:16]+' '+line_edit[17:]
                    else:
                        # Don't take the second and following alternate locations
                        continue

                residue_buffer.append(line_edit)


        if residue_buffer != []: # is there a residue in the buffer ?
            if not self.check_and_print_pdb(count, residue_buffer, residue_letter):
                # if unsuccessful
                self.shit_stat_misdns = True
            else:
                count = count + 1

        nres = len("".join(list(self.fastaseq.values())))

        flag_successful = True
        if nres <= 0:
            flag_successful = False

        flags = {
            "Alternative Positions" : self.shit_stat_altpos, 
            "Inserted Residue" : self.shit_stat_insres, 
            "Modified Residue" : self.shit_stat_modres, 
            "Missing backbone" : self.shit_stat_misdns, 
            "Successful" : flag_successful
            }

        fastaseqs = []
        if nres > 0:
            self.pdbfile += "TER\n"

            if not allchains:
                for chain in self.fastaseq:
                    fastaseqs.append(self.fastaseq[chain])
            else:
                fastaseqs = ["".join(list(self.fastaseq.values()))]

        # this only happens if files were downloaded
        if len(self.files_to_unlink) > 0:
            for file in self.files_to_unlink:
                os.unlink(file)

        return self.pdbfile, fastaseqs, nres, flags