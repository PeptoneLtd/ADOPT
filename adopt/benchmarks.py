# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import scipy
from plotly.subplots import make_subplots
import time

from adopt import CheZod, StabilityAnalysis, constants, utils

pio.renderers = "pdf"


def create_parser():
    parser = argparse.ArgumentParser(description="Run benchmarks")

    parser.add_argument(
        "benchmark_data_path",
        type=str,
        help="Path of the predictions made by the benchmark method on the test set",
    )
    parser.add_argument(
        "train_json_file",
        type=str,
        help="JSON file containing the proteins we want to use as training set",
    )
    parser.add_argument(
        "test_json_file",
        type=str,
        help="JSON file containing the proteins we want to use as test set",
    )
    parser.add_argument(
        "train_repr_dir",
        type=str,
        help="Training set residue level representation directory",
    )
    parser.add_argument(
        "test_repr_dir",
        type=str,
        help="Test set residue level representation directory",
    )
    parser.add_argument(
        "--train_strategy",
        type=str,
        metavar="",
        required=True,
        help="Training strategies",
    )
    parser.add_argument(
        "-a",
        "--msa",
        action="store_true",
        help="Extract MSA based representations",
    )
    return parser


def plot_corr_per_residue(corr_per_res, model_picked):
    # residue specific correlations
    sorted_dict = dict(
        sorted(
            corr_per_res[model_picked].items(),
            key=lambda item: item[1][1],
            reverse=True,
        )
    )
    res_types = [k for k in sorted_dict.keys()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=res_types,
            y=[v[1] for v in sorted_dict.values()],
            name=f"{model_picked}",
            marker_color="indianred",
        )
    )
    fig.add_trace(
        go.Bar(
            x=res_types,
            y=[corr_per_res["combined"][k][1] for k in res_types],
            name="combined-esm",
            marker_color="maroon",
        )
    )
    fig.add_trace(
        go.Bar(
            x=res_types,
            y=[corr_per_res["odin"][k][1] for k in res_types],
            name="odin-pred",
            marker_color="lightsalmon",
        )
    )
    fig.add_hline(y=0.64, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(
        # title="Correlations between predicted vs. actual z-scores per residue type",
        xaxis_title="Residue type",
        yaxis_title=r"$\rho_{\mathrm{Spearman}}$",
        # legend_title="Legend Title",
        font=dict(family="Courier New", size=16, color="black"),
    )
    fig.update_layout(
        barmode="group",
        legend=dict(
            orientation="h",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.46,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    pio.write_image(
        fig,
        "media/correlations_per_res_esm_odin.pdf",
        width=900,
        height=450,
        scale=1.0
    )
    time.sleep(5)
    pio.write_image(
        fig,
        "media/correlations_per_res_esm_odin.pdf",
        width=900,
        height=450,
        scale=1.0
    )


def plot_gt_vs_pred_contours(actual_z_scores, z_scores_per_model):
    # ESM-1b vs. ODiNPred - 1 plot
    # -------------------
    esm_z_scores = np.concatenate(z_scores_per_model["esm-1b"], axis=0)
    odin_z_scores = z_scores_per_model["odin"]

    def plotter(gt_scores, pred_scores, title='Title', ref_y_axes=esm_z_scores):
        fig = go.Figure()
        fig.add_trace(go.Histogram2dContour(
            x = gt_scores,
            y = pred_scores,
            colorscale = 'Blues',
            contours_coloring = 'heatmap',
            showscale=False,
            ))
        fig.add_shape(type="line",
            x0=min(gt_scores), y0=min(gt_scores),
            x1=max(gt_scores), y1=max(gt_scores),
            line=dict(
                color="darkred",
                width=2,
                dash="dashdot",
            ))
        fig.add_shape(type="line",
            x0=min(gt_scores), y0=min(gt_scores)+5,
            x1=max(gt_scores), y1=max(gt_scores)+5,
            line=dict(color="darkred",
                      width=1,
                      dash="dash",
                     )
                    )
        fig.add_shape(type="line",
                      x0=min(gt_scores), y0=min(gt_scores)-5,
                      x1=max(gt_scores), y1=max(gt_scores)-5,
                      line=dict(color="darkred",
                                width=1,
                                dash="dash",
                                )
                     )

        fig.update_yaxes(range=[min(ref_y_axes)-1, max(ref_y_axes)+1],  showgrid=False)
        fig.update_xaxes(range=[min(gt_scores)-1, max(gt_scores)+1],  showgrid=False)
        fig.update_layout(title=title,
                          xaxis_title="Experimental Z-scores",
                          yaxis_title="Predicted Z-scores",
                          legend_title="Legend Title",
                          font=dict(family="Courier New",
                                    size=18,
                                    color="black"
                                    )
                         )

        fig.update_traces(xbins=dict(start=-10., end=23.))
        fig.update_traces(ybins=dict(start=-10., end=23.))
        fig.update_layout(title_x=0.5)

        return fig

    fig_odin = plotter(actual_z_scores, odin_z_scores, title='ODiNPred')
    fig_esm = plotter(actual_z_scores, esm_z_scores, title='ESM')

    pio.write_image(fig_odin, f"media/odinpred_contours_with_ref.pdf", width=500, height=500, scale=1.)
    time.sleep(5)
    pio.write_image(fig_odin, f"media/odinpred_contours_with_ref.pdf", width=500, height=500, scale=1.)
    time.sleep(5)
    pio.write_image(fig_esm, f"media/esm1b_contours_with_ref.pdf", width=500, height=500, scale=1.)
    time.sleep(5)
    pio.write_image(fig_esm, f"media/esm1b_contours_with_ref.pdf", width=500, height=500, scale=1.)


class DisorderCompare:
    def __init__(self,
                 path_odinpred_examples,
                 path_chezod_1325_raw,
                 path_chezod_117_raw,
                 path_chezod_1325_repr,
                 path_chezod_117_repr,
                 model_types):
        self.path_odinpred_examples = path_odinpred_examples
        self.path_chezod_1325_raw = path_chezod_1325_raw
        self.path_chezod_117_raw = path_chezod_117_raw
        self.path_chezod_1325_repr = path_chezod_1325_repr
        self.path_chezod_117_repr = path_chezod_117_repr
        self.model_types = model_types
        if self.model_types == constants.msa_model_types:
            self.msa = True
        else:
            self.msa = False

    def get_z_score_per_residue(self, strategy):
        # read the data
        f_names_op_117 = next(os.walk(self.path_odinpred_examples))[2]
        f_names_op_117 = [file for file in f_names_op_117 if file.endswith(".txt")]
        chezod = CheZod(self.path_chezod_1325_raw, self.path_chezod_117_raw, self.model_types)
        _, _, df_117 = chezod.get_chezod_raw()

        predicted_z_scores = {
            "esm-1v": {},
            "esm-1b": {},
            "esm-msa": {},
            "combined": {},
            "odin": {},
        }

        for file_name in f_names_op_117:
            brmid_dummy = file_name.split(".")[0][len("DisorderPredictions"):]  # extract numbers in a better way

            # read the ODinPred txt file
            df_odin = pd.read_csv(
                f"{self.path_odinpred_examples}{file_name}", delim_whitespace=True
            )

            # get the original z-scores
            # get the sequence and the original z-scores and the esm-representations
            seq = list(df_117[df_117["brmid"] == brmid_dummy]["sequence"].item())
            zex = list(df_117[df_117["brmid"] == brmid_dummy]["zscore"].item())

            for model_type in constants.model_types:
                if model_type == "esm-msa":
                    msa_ind = True
                else:
                    msa_ind = False

                repr_path = utils.representation_path(
                    self.path_chezod_1325_repr, self.path_chezod_117_repr, self.msa
                )
                # get the representations and the experimental z_scores
                ex_dum, _ = utils.pedestrian_input(
                    [brmid_dummy],
                    df_117,
                    repr_path[model_type]["117"],
                    z_col="zscore",
                    msa=msa_ind,
                    drop_missing=False,
                )

                onnx_model = (
                    "models/lasso_"
                    + model_type
                    + "_"
                    + constants.strategies_dict[strategy]
                    + ".onnx"
                )

                # get the predictions for a given model
                for i in [
                    x
                    for x in zip(seq, zex, utils.get_onnx_model_preds(onnx_model, ex_dum))
                    if x[1] != 999.0
                ]:
                    if i[0] in predicted_z_scores[model_type].keys():
                        predicted_z_scores[model_type][i[0]].append(
                            [i[1], i[2], i[2] - i[1]]
                        )
                    else:
                        predicted_z_scores[model_type][i[0]] = [[i[1], i[2], i[2] - i[1]]]

            # combined output
            ex_dum_esm1v, _ = utils.pedestrian_input(
                [brmid_dummy],
                df_117,
                repr_path["esm-1v"]["117"],
                z_col="zscore",
                msa=False,
                drop_missing=False,
            )

            ex_dum_esm1b, _ = utils.pedestrian_input(
                [brmid_dummy],
                df_117,
                repr_path["esm-1b"]["117"],
                z_col="zscore",
                msa=False,
                drop_missing=False,
            )

            # the right order to concatenate - esm-1v and then esm-1b.
            # as this was used to fit the regression model
            ex_comb = np.concatenate((ex_dum_esm1v, ex_dum_esm1b), axis=1)

            comb_onnx_model = (
                "models/lasso_"
                + "combined"
                + "_"
                + constants.strategies_dict[strategy]
                + ".onnx"
            )

            for i in [
                x
                for x in zip(seq, zex, utils.get_onnx_model_preds(comb_onnx_model, ex_comb))
                if x[1] != 999.0
            ]:
                if i[0] in predicted_z_scores["combined"].keys():
                    predicted_z_scores["combined"][i[0]].append([i[1], i[2], i[2] - i[1]])
                else:
                    predicted_z_scores["combined"][i[0]] = [[i[1], i[2], i[2] - i[1]]]

            for ii in [x for x in zip(seq, zex, list(df_odin["Zscore"])) if x[1] != 999.0]:
                if ii[0] in predicted_z_scores["odin"].keys():
                    predicted_z_scores["odin"][ii[0]].append([ii[1], ii[2], ii[2] - ii[1]])
                else:
                    predicted_z_scores["odin"][ii[0]] = [[ii[1], ii[2], ii[2] - ii[1]]]
        return predicted_z_scores


class CheZodCompare:
    def __init__(self, predicted_z_scores):
        self.predicted_z_scores = predicted_z_scores

    def get_corr_per_residue(self):
        corr_per_res = {
            "esm-1v": {},
            "esm-1b": {},
            "esm-msa": {},
            "combined": {},
            "odin": {},
        }
        for model_type in corr_per_res.keys(): # todo parallelise this loop
            for k in self.predicted_z_scores[model_type].keys():
                print("residue type: ", k)
                print("--------------------------------")
                print("number - ", len(self.predicted_z_scores[model_type][k]))
                print(
                    "mean - ground truth/predicted: ",
                    np.mean([x[0] for x in self.predicted_z_scores[model_type][k]]),
                    np.mean([x[1] for x in self.predicted_z_scores[model_type][k]]),
                )
                print(
                    "correlation between GT and pred: ",
                    scipy.stats.spearmanr(
                        [x[0] for x in self.predicted_z_scores[model_type][k]],
                        [x[1] for x in self.predicted_z_scores[model_type][k]],
                    ).correlation,
                )
                corr_per_res[model_type][k] = [
                    len(self.predicted_z_scores[model_type][k]),
                    scipy.stats.spearmanr(
                        [x[0] for x in self.predicted_z_scores[model_type][k]],
                        [x[1] for x in self.predicted_z_scores[model_type][k]],
                    ).correlation,
                ]
                print()
        return corr_per_res

    def get_z_scores_per_model(self):
        actual_z_scores = []
        z_scores_per_model = {
            model_type: [] for model_type in self.predicted_z_scores.keys()
        }
        for model_type in z_scores_per_model.keys():
            for key in self.predicted_z_scores[model_type].keys():
                for i in range(len(self.predicted_z_scores[model_type][key])):
                    z_scores_per_model[model_type].append(
                        self.predicted_z_scores[model_type][key][i][1]
                    )
                    if model_type == "odin":
                        actual_z_scores.append(
                            self.predicted_z_scores[model_type][key][i][0]
                        )
        return actual_z_scores, z_scores_per_model


def main(args):
    if args.train_strategy not in constants.train_strategies:
        print("The training strategies are:")
        print(*constants.train_strategies, sep="\n")
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
    model_types = utils.get_model_types(args.msa)
    disorder_compare = DisorderCompare(args.benchmark_data_path,
                                       args.train_json_file,
                                       args.test_json_file,
                                       args.train_repr_dir,
                                       args.test_repr_dir,
                                       model_types)
    stability_analysis = StabilityAnalysis(args.train_json_file,
                                           args.test_json_file,
                                           args.train_repr_dir,
                                           args.test_repr_dir,
                                           model_types)

    predicted_z_scores = disorder_compare.get_z_score_per_residue(args.train_strategy)
    chezod_compare = CheZodCompare(predicted_z_scores)
    corr_per_res = chezod_compare.get_corr_per_residue()
    actual_z_scores, z_scores_per_model = chezod_compare.get_z_scores_per_model()
    plot_gt_vs_pred_contours(actual_z_scores, z_scores_per_model)

    for model_picked in constants.model_types:
        probas = stability_analysis.get_stability_paths(model_picked)
        stability_analysis.plot_stability_paths(probas, model_picked)
        plot_corr_per_residue(corr_per_res, model_picked)
