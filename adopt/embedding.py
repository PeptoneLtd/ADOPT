# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import subprocess
from pathlib import Path

from adopt import constants


def create_parser():
    parser = argparse.ArgumentParser(
        description="Extract residue level representations"
    )

    parser.add_argument(
        "-f",
        "--fasta_path",
        type=str,
        metavar="",
        required=True,
        help="FASTA file containing the proteins for which you want to compute the intrinsic disorder",
    )
    parser.add_argument(
        "-r",
        "--repr_dir",
        type=str,
        metavar="",
        required=True,
        help="Residue level representation directory",
    )
    return parser


# extract residue level representations of each protein sequence in the fasta file
def get_representations(fasta_file, repr_dir):
    for esm_model in constants.esm_models:
        model_dir = str(repr_dir) + "/" + constants.models_dict[esm_model]
        Path(str(model_dir)).mkdir(parents=True, exist_ok=True)
        if "esm_msa" in esm_model:
            bashCommand = (
                "python ../esm/extract.py "
                + str(esm_model)
                + " "
                + str(fasta_file)
                + " "
                + model_dir
                + " --repr_layers 12 --include per_tok"
            )  # todo fasta_file->msa_fasta_file
        else:
            bashCommand = (
                "python ../esm/extract.py "
                + str(esm_model)
                + " "
                + str(fasta_file)
                + " "
                + model_dir
                + " --repr_layers 33 --include per_tok"
            )
        process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
        output, error = process.communicate()


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    get_representations(args.fasta_path, args.repr_dir)
