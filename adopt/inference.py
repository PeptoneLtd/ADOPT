# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from adopt import constants, utils


def create_parser():
    parser = argparse.ArgumentParser(description="Predict intrinsic disorder (Z score)")

    parser.add_argument(
        "fasta_path",
        type=str,
        help="FASTA file containing the proteins for which you want to compute the intrinsic disorder",
    )
    parser.add_argument(
        "repr_dir",
        type=str,
        help="Residue level representation directory",
    )
    parser.add_argument(
        "pred_z_scores_path",
        type=str,
        help="Path where you want the Z scores to be saved",
    )
    parser.add_argument(
        "--train_strategy",
        type=str,
        choices=constants.train_strategies,
        help="Training strategies",
        required=True
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=constants.msa_model_types,
        help="pre-trained model we want to use",
        required=True
    )
    return parser


class ZScorePred:
    def __init__(self, strategy='train_on_total', model_type='esm-1b'):
        self.strategy = strategy
        self.model_type = model_type

        model_path = f"models/lasso_{self.model_type}_{constants.strategies_dict[self.strategy]}.onnx"

        model_path_file = Path(model_path)
        if model_path_file.is_file():
            print("Loading model file")
        else:
            bashCommand = (
                "wget https://adopt-models.s3.eu-west-2.amazonaws.com/models.zip"
                + " && "
                + "unzip models.zip"
            )
            process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
            output, error = process.communicate()

        self.onnx_model = (model_path)

    def get_z_score(self, representation):
        predicted_z_scores = utils.get_onnx_model_preds(
            self.onnx_model, representation.squeeze().numpy()
        )
        return np.concatenate(predicted_z_scores)

    def get_z_score_from_fasta(
        self,
        inference_fasta_path,
        inference_repr_path,
        predicted_z_scores_path,
    ):
        df_fasta = utils.fasta_to_df(inference_fasta_path)

        if self.model_type == "combined":
            repr_path = inference_repr_path + "/" + "esm-1v"
        else:
            repr_path = inference_repr_path + "/" + self.model_type

        repr_files = os.listdir(repr_path)
        indexes = []

        for file in repr_files:
            indexes.append(file.split(".")[0])

        predicted_z_scores = []

        for ix in indexes:
            if self.model_type == "esm-msa":
                repr_esm = (
                    torch.load(str(repr_path) + "/" + ix + ".pt")["representations"]
                    .clone()
                    .cpu()
                    .detach()
                )
            elif self.model_type == "combined":
                esm1b_repr_path = inference_repr_path + "/" + "esm-1b"
                repr_esm1v = (
                    torch.load(str(repr_path) + "/" + ix + ".pt")["representations"][33]
                    .clone()
                    .cpu()
                    .detach()
                )
                repr_esm1b = (
                    torch.load(str(esm1b_repr_path) + "/" + ix + ".pt")[
                        "representations"
                    ][33]
                    .clone()
                    .cpu()
                    .detach()
                )
                repr_esm = torch.cat([repr_esm1v, repr_esm1b], 1)
            else:
                repr_esm = (
                    torch.load(str(repr_path) + "/" + ix + ".pt")["representations"][33]
                    .clone()
                    .cpu()
                    .detach()
                )
            z_scores = utils.get_onnx_model_preds(self.onnx_model, repr_esm.numpy())
            predicted_z_scores.append(np.concatenate(z_scores))

        df_z = pd.DataFrame({"brmid": indexes, "z_scores": predicted_z_scores})
        df_results = df_fasta.join(df_z.set_index("brmid"), on="brmid")
        df_results.to_json(predicted_z_scores_path, orient="records")


def main(args):
    if args.train_strategy not in constants.train_strategies:
        print("The training strategies are:")
        print(*constants.train_strategies, sep="\n")
        sys.exit(2)

    if (args.model_type not in constants.msa_model_types) and (
        args.model_type != "combined"
    ):
        print("The pre-trained models are:")
        print(*constants.msa_model_types, sep="\n")
        print("combined")
        sys.exit(2)

    if (args.train_strategy != "train_on_cleared_1325_test_on_117_residue_split") and (
        args.model_type == "combined"
    ):
        print(
            "Only the train_on_cleared_1325_test_on_117_residue_split strategy"
            "is allowed with the <combined> model"
        )
        sys.exit(2)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    z_score_pred = ZScorePred(args.mode, args.train_strategy, args.model_type)
    z_score_pred.get_z_score_from_fasta(
        args.fasta_path, args.repr_dir, args.pred_z_scores_path
    )
