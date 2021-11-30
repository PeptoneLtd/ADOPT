import os
import argparse
import string
from collections import OrderedDict
import itertools
import torch
import esm
from Bio import SeqIO
from adopt import constants
from adopt import utils

# Create the parser
my_parser = argparse.ArgumentParser(description='Get the path to .a3m files for the sequences in the fasta file')

my_parser.add_argument('fasta_file',
                       action='store',
                       type=str)
my_parser.add_argument('repr_path',
                       action='store',
                       type=str,
                       help='path to save the esm-msa representations')

# Execute the parse_args() method
args = my_parser.parse_args()

a3m_input_path = '/msas/'  # args.msa_path
ff_path = args.fasta_file
repr_path = args.repr_path
DEFAULT_MSA_DEPTH = constants.msa_depth  # args.msa_depth

# Get the identifiers from the fasta file
df_msas = utils.fasta_to_df(ff_path)

# Check if the msas are available for the sequences provided in the fasta file
# ----------------------------------------------------------------------------
msa_store = list(next(os.walk(a3m_input_path))[2])
missing_msas = [seq for seq in list(df_msas['brmid']) if seq not in msa_store]
if len(missing_msas) != 0:
    print("No msa files were found for the following sequences: ", missing_msas)
    print("Generate msa files first!")

# download the esm-msa model
# --------------------------
msa_transformer, msa_alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S()
msa_transformer = msa_transformer.eval()
msa_batch_converter = msa_alphabet.get_batch_converter()

# Utilities for esm model
# -------------------------
# This is an efficient way to delete lowercase characters and insertion characters from a string
deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)

def read_sequence(filename: str):
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)


def remove_insertions(sequence: str):
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)


def read_msa(filename: str, nseq: int):
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq)))
            for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)]


msas_data = list(map(read_msa, [a3m_input_path+f_name for f_name in list(df_msas['brmid'])],
                     len(list(df_msas['brmid']))*[DEFAULT_MSA_DEPTH]))

# saving esm_msa representations
# ------------------------------
for protein_id in msas_data:
    msa_data = [protein_id]
    msa_batch_labels, msa_batch_strs, msa_batch_tokens = msa_batch_converter(msa_data)
    # msa_batch_tokens = msa_batch_tokens.cuda()

    with torch.no_grad():
        results = msa_transformer(msa_batch_tokens, repr_layers=[12], return_contacts=True)

    # saving
    sv = OrderedDict()
    sv['logits'] = results['logits'][0][0][1: , ...].clone()
    sv['representations'] = results['representations'][12][0][0][1: , ...].clone()
    sv['contacts'] = results['contacts'][0].clone()

    print(f'Saving - {msa_batch_labels[0][0]}.pt')
    torch.save(sv, f'{repr_path}{msa_batch_labels[0][0]}.pt')
