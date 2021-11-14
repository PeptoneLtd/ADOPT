# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys
import getopt
from pathlib import Path
from adopt import constants

# extract residue level representations of each protein sequence in the fasta file
def get_representations(fasta_file, repr_dir):
    for esm_model in constants.esm_models:
        model_dir = str(repr_dir)+"/"+constants.models_dict[esm_model]
        Path(str(model_dir)).mkdir(parents=True, exist_ok=True)
        if 'esm_msa' in esm_model:
            bashCommand = "python ../esm/extract.py "+str(esm_model)+" "+str(fasta_file)+" "+model_dir+ " --repr_layers 12 --include per_tok" # todo fasta_file->msa_fasta_file
        else:
            bashCommand = "python ../esm/extract.py "+str(esm_model)+" "+str(fasta_file)+" "+model_dir+ " --repr_layers 33 --include per_tok"
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hf:r:", ["fasta_file=", "repr_dir="]) 
    except getopt.GetoptError:
        print('usage: embedding.py -f <fasta_file_path=> -r <residue_level_representation_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: embedding.py -f <fasta_files_dir> -r <residue_level_representation_dir>')
            sys.exit()
        elif opt in ("-f", "--fasta_dir"):
            fasta_dir = arg
        elif opt in ("-r", "--repr_dir"):
            repr_dir = arg

    get_representations(fasta_dir, repr_dir)

if __name__ == "__main__":
    main(sys.argv[1:])