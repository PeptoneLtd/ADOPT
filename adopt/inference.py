# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

from adopt import constants, utils


def create_parser():
    parser = argparse.ArgumentParser(
        description="Predict intrinsic disorder (Z score)"
    )

    parser.add_argument(
        "-s",
        "--train_strategy",
        type=str,
        metavar="",
        required=True,
        help="Training strategies",
    )
    parser.add_argument(
        "-m",
        "--model_type",
        type=str,
        metavar="",
        required=True,
        help="pre-trained model we want to use",
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
    parser.add_argument(
        "-p",
        "--pred_z_scores_path",
        type=str,
        metavar="",
        required=True,
        help="Path where you want the Z scores to be saved",
    )
    parser.add_argument(
        "-a",
        "--msa",
        action="store_true",
        help="Extract MSA based representations",
    )
    return parser


class ZScorePred:
    def __init__(self, strategy, model_type):
        self.strategy = strategy
        self.model_type = model_type
        self.onnx_model = (
            "../models/lasso_"
            + self.model_type
            + "_"
            + constants.strategies_dict[self.strategy]
            + ".onnx"
        )

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
                    torch.load(str(repr_path) + "/" + ix + ".pt")["representations"][12]
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
    model_types = utils.get_model_types(args.msa)
    if args.train_strategy not in constants.train_strategies:
        print("The training strategies are:")
        print(*constants.train_strategies, sep="\n")
        sys.exit(2)

    if (args.model_type not in model_types) and (
        args.model_type != "combined"
    ):
        print("The pre-trained models are:")
        print(*model_types, sep="\n")
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
    z_score_pred = ZScorePred(args.train_strategy, args.model_type)
    z_score_pred.get_z_score_from_fasta(
        args.fasta_path, args.repr_dir, args.pred_z_scores_path
    )
