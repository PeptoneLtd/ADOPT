# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys

import numpy as np
import scipy
from sklearn import linear_model
from sklearn.model_selection import KFold

from adopt import CheZod, constants, utils


# disorder predictor training
def create_parser():
    parser = argparse.ArgumentParser(description="Train ADOPT")

    parser.add_argument(
        "-s",
        "--train_strategy",
        type=str,
        metavar="",
        required=True,
        help="Training strategies",
    )
    parser.add_argument(
        "-t",
        "--train_json_file",
        type=str,
        metavar="",
        required=True,
        help="JSON file containing the proteins we want to use as training set",
    )
    parser.add_argument(
        "-e",
        "--test_json_file",
        type=str,
        metavar="",
        required=True,
        help="JSON file containing the proteins we want to use as test set",
    )
    parser.add_argument(
        "-r",
        "--train_repr_dir",
        type=str,
        metavar="",
        required=True,
        help="Training set residue level representation directory",
    )
    parser.add_argument(
        "-p",
        "--test_repr_dir",
        type=str,
        metavar="",
        required=True,
        help="Test set residue level representation directory",
    )
    parser.add_argument(
        "-a",
        "--msa",
        action="store_true",
        help="Extract MSA based representations",
    )
    return parser


class DisorderPred:
    def __init__(
        self,
        path_chezod_1325_raw,
        path_chezod_117_raw,
        path_chezod_1325_repr,
        path_chezod_117_repr,
        model_types,
    ):
        self.path_chezod_1325_raw = str(path_chezod_1325_raw)
        self.path_chezod_117_raw = str(path_chezod_117_raw)
        self.path_chezod_1325_repr = str(path_chezod_1325_repr)
        self.path_chezod_117_repr = str(path_chezod_117_repr)
        self.model_types = model_types
        if self.model_types == constants.msa_model_types:
            self.msa = True
        else:
            self.msa = False
        chezod = CheZod(self.path_chezod_1325_raw, self.path_chezod_117_raw, self.model_types)
        (
            self.ex_train,
            self.zed_train,
            self.ex_test,
            self.zed_test,
        ) = chezod.get_train_test_sets(
            self.path_chezod_1325_repr, self.path_chezod_117_repr
        )
        _, self.df_ch, _ = chezod.get_chezod_raw()
        self.repr_path = utils.representation_path(
            self.path_chezod_1325_repr, self.path_chezod_117_repr, self.msa
        )

    def cleared_residue(self):
        # residue level split, train on cleared chezod 1325 and validation on chezod 117
        CorrelationsLR = {}
        LinearRegressions = {}

        for model_type in self.model_types:
            reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
            reg_.fit(self.ex_train[model_type], self.zed_train[model_type])

            print(
                f"{model_type} - Correlation between the predicted and the ground trouth on the test set: ",
                scipy.stats.spearmanr(
                    self.zed_test[model_type], reg_.predict(self.ex_test[model_type])
                ).correlation,
            )

            LinearRegressions[model_type] = reg_
            CorrelationsLR[model_type] = scipy.stats.spearmanr(
                self.zed_test[model_type], reg_.predict(self.ex_test[model_type])
            ).correlation
            utils.save_onnx_model(
                self.ex_test[model_type].shape[1],
                reg_,
                "../models/lasso_" + model_type + "_cleared_residue.onnx",
            )

        # ESM-1v and ESM-1b Combined
        # --------------------------
        ex_train_combined = np.concatenate(
            (self.ex_train["esm-1v"], self.ex_train["esm-1b"]), axis=1
        )
        ex_test_combined = np.concatenate(
            (self.ex_test["esm-1v"], self.ex_test["esm-1b"]), axis=1
        )

        reg = linear_model.Lasso(alpha=0.0001, max_iter=10000)
        reg.fit(ex_train_combined, self.zed_train["esm-1v"])

        print(
            "Combining esm-1v and esm-1b - Correlation between the predicted and the ground trouth on the test set: ",
            scipy.stats.spearmanr(
                self.zed_test["esm-1v"], reg.predict(ex_test_combined)
            ).correlation,
        )

        # save the combined regression
        LinearRegressions["combined"] = reg
        utils.save_onnx_model(
            ex_test_combined.shape[1],
            reg,
            "../models/lasso_combined_cleared_residue.onnx",
        )

    def residue_cv(self):
        # assemble the training data from the 1325 set
        # ex_1325, zed_1325 = pedestrian_input(list(df_ch['brmid']), df_ch, path_chezod_esm_repr, z_col='z-score')
        # read the data
        ex_1325, zed_1325 = {}, {}

        for model_type in self.model_types:
            if model_type == "esm-msa":
                msa_ind = True
            else:
                msa_ind = False

            # assemble the training data from the 1325 set
            ex_1325[model_type], zed_1325[model_type] = utils.pedestrian_input(
                list(self.df_ch["brmid"]),
                self.df_ch,
                self.repr_path[model_type]["1325"],
                z_col="z-score",
                msa=msa_ind,
            )

        # 10 fold CV on the 1325 set
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(
            ex_1325[self.model_types[0]]
        )  # since the number of inputs are the same for all 3 models,
        # and the splits are based on indices, it's okay to select indices based only
        # on one of the model types
        corrs = {model_type: [] for model_type in self.model_types}
        regressors = {model_type: [] for model_type in self.model_types}
        rounds = 1

        for train_index, test_index in kf.split(ex_1325[self.model_types[0]]):
            print("rounds: ", rounds)
            print("-----------------")

            for model_type in self.model_types:
                print(model_type)
                ex_rounds_train = np.take(ex_1325[model_type], train_index, axis=0)
                ex_rounds_test = np.take(ex_1325[model_type], test_index, axis=0)

                zed_rounds_train = np.take(zed_1325[model_type], train_index, axis=0)
                zed_rounds_test = np.take(zed_1325[model_type], test_index, axis=0)

                reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
                reg_.fit(ex_rounds_train, zed_rounds_train)

                corrs[model_type].append(
                    scipy.stats.spearmanr(
                        zed_rounds_test, reg_.predict(ex_rounds_test)
                    ).correlation
                )
                regressors[model_type].append(reg_)

                print(
                    "Correlation between the predicted and the ground trouth on the test set: ",
                    scipy.stats.spearmanr(
                        zed_rounds_test, reg_.predict(ex_rounds_test)
                    ).correlation,
                )
                print()
            rounds += 1

        for model_type in self.model_types:
            # save best regressor for inference
            index_min_corr = min(
                range(len(corrs[model_type])), key=corrs[model_type].__getitem__
            )
            best_reg = regressors[model_type][index_min_corr]
            utils.save_onnx_model(
                ex_rounds_test.shape[1],
                best_reg,
                "../models/lasso_" + model_type + "_residue_cv.onnx",
            )

            print(model_type)
            print(
                "10-fold CV - average correlation between the predicted and the ground trouth on the test set: ",
                np.mean(corrs[model_type]),
            )
            print()

    def cleared_residue_cv(self):
        # 10 fold CV on the cleared 1325 set, i.e. removing the proteins that appear also in the 117 set
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(self.ex_train[self.model_types[0]])

        corrs_cleared = {model_type: [] for model_type in self.model_types}
        regressors = {model_type: [] for model_type in self.model_types}
        rounds = 1

        for train_index, test_index in kf.split(
            self.ex_train[self.model_types[0]]
        ):
            print("rounds: ", rounds)
            print("-----------------")

            for model_type in self.model_types:
                ex_rounds_train = np.take(
                    self.ex_train[model_type], train_index, axis=0
                )
                ex_clear_rounds_test = np.take(
                    self.ex_train[model_type], test_index, axis=0
                )

                zed_rounds_train = np.take(
                    self.zed_train[model_type], train_index, axis=0
                )
                zed_rounds_test = np.take(
                    self.zed_train[model_type], test_index, axis=0
                )

                reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
                reg_.fit(ex_rounds_train, zed_rounds_train)

                corrs_cleared[model_type].append(
                    scipy.stats.spearmanr(
                        zed_rounds_test, reg_.predict(ex_clear_rounds_test)
                    ).correlation
                )
                regressors[model_type].append(reg_)

                print(
                    "Correlation between the predicted and the ground trouth on the test set: ",
                    scipy.stats.spearmanr(
                        zed_rounds_test, reg_.predict(ex_clear_rounds_test)
                    ).correlation,
                )

                print()
            rounds += 1

        for model_type in self.model_types:
            # save best regressor for inference
            index_min_corr = min(
                range(len(corrs_cleared[model_type])),
                key=corrs_cleared[model_type].__getitem__,
            )
            best_reg = regressors[model_type][index_min_corr]
            utils.save_onnx_model(
                ex_clear_rounds_test.shape[1],
                best_reg,
                "../models/lasso_" + model_type + "_cleared_residue_cv.onnx",
            )

            print(model_type)
            print("10-fold CV (on the reduced 1325, i.e. overlap removed) - ")
            print(
                "average correlation between the predicted and the ground trouth on the test set: ",
                np.mean(corrs_cleared[model_type]),
            )

    def cleared_sequence_cv(self):
        # 10-fold CV protein sequence based fold selection applied on cleared chezod 1325
        seq_ids = list(self.df_ch["brmid"])

        # 10 fold CV on the cleared 1325 set, i.e. removing the proteins that appear also in the 117 set
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(seq_ids)

        corrs_seq = {model_type: [] for model_type in self.model_types}
        regressors = {model_type: [] for model_type in self.model_types}
        rounds_seq = 1

        for train_index, test_index in kf.split(seq_ids):
            print("rounds: ", rounds_seq)
            print("-----------------")

            for model_type in self.model_types:

                if model_type == "esm-msa":
                    msa_ind = True
                else:
                    msa_ind = False

                # assemble the training data from the cleared 1325 set
                train_brmids = np.take(seq_ids, train_index, axis=0)
                test_brmids = np.take(seq_ids, test_index, axis=0)

                ex_train_seq, zed_train_seq = utils.pedestrian_input(
                    train_brmids,
                    self.df_ch,
                    self.repr_path[model_type]["1325"],
                    z_col="z-score",
                    msa=msa_ind,
                    drop_missing=True,
                )
                ex_test_seq, zed_test_seq = utils.pedestrian_input(
                    test_brmids,
                    self.df_ch,
                    self.repr_path[model_type]["1325"],
                    z_col="z-score",
                    msa=msa_ind,
                    drop_missing=True,
                )

                reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
                reg_.fit(ex_train_seq, zed_train_seq)

                corrs_seq[model_type].append(
                    scipy.stats.spearmanr(
                        zed_test_seq, reg_.predict(ex_test_seq)
                    ).correlation
                )
                regressors[model_type].append(reg_)

                print(
                    "Correlation between the predicted and the ground trouth on the test set: ",
                    scipy.stats.spearmanr(
                        zed_test_seq, reg_.predict(ex_test_seq)
                    ).correlation,
                )

                print()
            rounds_seq += 1

        for model_type in self.model_types:
            # save best regressor for inference
            index_min_corr = min(
                range(len(corrs_seq[model_type])), key=corrs_seq[model_type].__getitem__
            )
            best_reg = regressors[model_type][index_min_corr]
            utils.save_onnx_model(
                ex_test_seq.shape[1],
                best_reg,
                "../models/lasso_" + model_type + "_cleared_sequence_cv.onnx",
            )

            print(model_type)
            print("10-fold CV - Folds split on sequence level")
            print(
                "average correlation between the predicted and the ground trouth on the test set: ",
                np.mean(corrs_seq[model_type]),
            )


def main(args):
    if args.train_strategy not in constants.train_strategies:
        print("The training strategies are:")
        print(*constants.train_strategies, sep="\n")
        sys.exit(2)


if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)
    model_types = utils.get_model_types(args.msa)
    disorder_pred = DisorderPred(
        args.train_json_file,
        args.test_json_file,
        args.train_repr_dir,
        args.test_repr_dir,
        model_types
    )
    if args.train_strategy == "train_on_cleared_1325_test_on_117_residue_split":
        disorder_pred.cleared_residue()
    elif args.train_strategy == "train_on_1325_cv_residue_split":
        disorder_pred.residue_cv()
    elif args.train_strategy == "train_on_cleared_1325_cv_residue_split":
        disorder_pred.cleared_residue_cv()
    else:
        disorder_pred.cleared_sequence_cv()
