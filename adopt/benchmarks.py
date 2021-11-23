# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import numpy as np
import pandas as pd
import scipy
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

from adopt import CheZod, constants, utils


def create_parser():
    parser = argparse.ArgumentParser(description="Run benchmarks")

    parser.add_argument(
        "-b",
        "--benchmark_data_path",
        type=str,
        metavar="",
        required=True,
        help="Path of the predictions made by the benchmark method on the test set",
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
        "-s",
        "--train_strategy",
        type=str,
        metavar="",
        required=True,
        help="Training strategies",
    )
    return parser


def get_z_score_per_residue(path_chezod_examples, 
                            path_chezod_1325_raw, 
                            path_chezod_117_raw, 
                            path_chezod_1325_repr, 
                            path_chezod_117_repr,
                            strategy):
    # read the data 
    f_names_op_117 = next(os.walk(path_chezod_examples))[2]
    f_names_op_117 = [file for file in f_names_op_117 if file.endswith(".txt")]
    chezod = CheZod(path_chezod_1325_raw, path_chezod_117_raw)
    _, _, df_117 = chezod.get_chezod_raw()

    predicted_z_scores = {'esm-1v': {}, 
                          'esm-1b': {},
                          'esm-msa': {},
                          'combined': {},
                          'odin': {}}

    for file_name in f_names_op_117:
        brmid_dummy = file_name.split('.')[0][len('DisorderPredictions'):] # extract numbers in a better way
        
        # read the ODinPred txt file
        df_odin = pd.read_csv(f'{path_chezod_examples}{file_name}', delim_whitespace=True)
        
        # get the original z-scores
        # get the sequence and the original z-scores and the esm-representations
        seq = list(df_117[df_117['brmid']==brmid_dummy]['sequence'].item())
        zex = list(df_117[df_117['brmid']==brmid_dummy]['zscore'].item())
        
        for model_type in constants.model_types:
            if model_type=="esm-msa":
                msa_ind=True
            else:
                msa_ind=False
            
            repr_path = utils.representation_path(path_chezod_1325_repr, path_chezod_117_repr)
            # get the representations and the experimental z_scores
            ex_dum, zed_dum = utils.pedestrian_input([brmid_dummy], df_117, repr_path[model_type]['117'], 
                                                    z_col='zscore', msa=msa_ind, drop_missing=False)
            
            onnx_model = (
            "../models/lasso_"
            + model_type
            + "_"
            + constants.strategies_dict[strategy]
            + ".onnx"
            )
            
            # get the predictions for a given model
            for i in [x for x in zip(seq, zex, utils.get_onnx_model_preds(onnx_model, ex_dum)) if x[1]!=999.]:
                if i[0] in predicted_z_scores[model_type].keys():
                    predicted_z_scores[model_type][i[0]].append([i[1], i[2], i[2]-i[1]])
                else:
                    predicted_z_scores[model_type][i[0]]=[[i[1], i[2], i[2]-i[1]]]
            
        # combined output 
        ex_dum_esm1v, zed_dum_esm1v = utils.pedestrian_input([brmid_dummy], df_117, repr_path['esm-1v']['117'], 
                                                    z_col='zscore', msa=False, drop_missing=False)
        
        ex_dum_esm1b, zed_dum_esm1b = utils.pedestrian_input([brmid_dummy], df_117, repr_path['esm-1b']['117'], 
                                                    z_col='zscore', msa=False, drop_missing=False)
        
        # the right order to concatenate - esm-1v and then esm-1b. 
        # as this was used to fit the regression model 
        ex_comb = np.concatenate((ex_dum_esm1v, ex_dum_esm1b), axis=1)
    
        comb_onnx_model = (
            "../models/lasso_"
            + 'combined'
            + "_"
            + constants.strategies_dict[strategy]
            + ".onnx"
            )

        for i in [x for x in zip(seq, zex, utils.get_onnx_model_preds(comb_onnx_model, ex_comb)) if x[1]!=999.]:
            if i[0] in predicted_z_scores['combined'].keys():
                predicted_z_scores['combined'][i[0]].append([i[1], i[2], i[2]-i[1]])
            else:
                predicted_z_scores['combined'][i[0]]=[[i[1], i[2], i[2]-i[1]]]
                
        for ii in [x for x in zip(seq, zex, list(df_odin['Zscore'])) if x[1]!=999.]:
            if ii[0] in predicted_z_scores['odin'].keys():
                predicted_z_scores['odin'][ii[0]].append([ii[1], ii[2], ii[2]-ii[1]])
            else:
                predicted_z_scores['odin'][ii[0]]=[[ii[1], ii[2], ii[2]-ii[1]]]

    return predicted_z_scores


def plot_corr_per_residue(corr_per_res, model_picked):
    # residue specific correlations
    sorted_dict = dict(sorted(corr_per_res[model_picked].items(), key=lambda item: item[1][1], reverse=True))
    res_types = [k for k in sorted_dict.keys()]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=res_types,
        y=[v[1] for v in  sorted_dict.values()],
        name=f'{model_picked}',
        marker_color='indianred'
    ))
    fig.add_trace(go.Bar(
        x=res_types,
        y=[corr_per_res['combined'][k][1] for k in res_types],
        name='combined-esm',
        marker_color='maroon'
    ))
    fig.add_trace(go.Bar(
        x=res_types,
        y=[corr_per_res['odin'][k][1] for k in res_types],
        name='odin-pred',
        marker_color='lightsalmon'
    ))
    fig.add_hline(y=0.64, line_width=3, line_dash="dash", line_color="green")
    fig.update_layout(
        #title="Correlations between predicted vs. actual z-scores per residue type",
        xaxis_title="Residue type",
        yaxis_title=r'$\rho_{\mathrm{Spearman}}$',
        #legend_title="Legend Title",
        font=dict(
            family="Courier New",
            size=16,
            color="black"
        )
    )
    fig.update_layout(barmode='group', 
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.46, 
                        bgcolor='rgba(0,0,0,0)'
    ))
    pio.write_image(fig, f"../media/correlations_per_res_esm_odin.pdf", width=900, height=450, scale=1.)


def plot_gt_vs_pred_contours(actual_z_scores, z_scores_per_model):
    # ESM-1b vs. ODiNPred - 1 plot 
    # -------------------
    fig = make_subplots(rows=1, cols=2, subplot_titles=("(ESM)", 
                                                        "(ODiNPred)"))
    esm_z_scores = z_scores_per_model['esm-1b']
    odin_z_scores = z_scores_per_model['odin']
    x = actual_z_scores
    y = esm_z_scores

    fig.add_trace(go.Histogram2dContour(
            x = actual_z_scores,
            y = esm_z_scores,
            colorscale = 'Blues',
            contours_coloring = 'heatmap',
            showscale=False,
            #reversescale = False,
            
        ), row=1, col=1)
    fig.add_shape(type="line",
        x0=min(actual_z_scores), y0=min(actual_z_scores), 
        x1=max(actual_z_scores), y1=max(actual_z_scores),
        line=dict(
            color="darkred",
            width=2,
            dash="dashdot",
        ), row=1, col=1)
    fig.add_trace(go.Histogram2dContour(
            x = actual_z_scores,
            y = odin_z_scores,
            colorscale = 'Blues',
            contours_coloring = 'heatmap',
            showscale=False,
            #reversescale = False,
        ), row=1, col=2)
    fig.add_shape(type="line",
        x0=min(actual_z_scores), y0=min(actual_z_scores), 
        x1=max(actual_z_scores), y1=max(actual_z_scores),
        line=dict(
            color="darkred",
            width=2,
            dash="dashdot",
        ), row=1, col=2)
    fig.update_xaxes(title_text="Experimental Z-scores", range=[min(actual_z_scores)-1, max(actual_z_scores)+1], row=1, col=1)
    fig.update_xaxes(title_text="Experimental Z-scores", range=[min(actual_z_scores)-1, max(actual_z_scores)+1], row=1, col=2)
    fig.update_yaxes(title_text="Predicted Z-scores",   range=[min(esm_z_scores)-1, max(esm_z_scores)+1], showgrid=False, row=1, col=1)
    fig.update_yaxes(title_text="Predicted Z-scores",   range=[min(esm_z_scores)-1, max(esm_z_scores)+1], showgrid=False, row=1, col=2)
    fig.update_traces(xbins=dict(start=-10., end=23.), row=1, col=2)
    fig.update_traces(ybins=dict(start=-10., end=23.), row=1, col=2)
    fig.update_layout( font=dict(
            family="Courier New",
            size=16,
            color="black"
        ))
    pio.write_image(fig, f"../media/esm1b_odinpred_contours_with_ref.pdf",  width=900, height=450, scale=1.)


class CheZodCompare:
    def __init__(self, predicted_z_scores):
        self.predicted_z_scores = predicted_z_scores

    def get_corr_per_residue(self):
        corr_per_res = {'esm-1v': {}, 
                        'esm-1b': {},
                        'esm-msa': {},
                        'combined': {},
                        'odin': {}}
        for model_type in corr_per_res.keys():
            for k in self.predicted_z_scores[model_type].keys():
                print('residue type: ', k)
                print('--------------------------------')
                print('number - ', len(self.predicted_z_scores[model_type][k]))
                print('mean - ground truth/predicted: ', np.mean([x[0] for x in self.predicted_z_scores[model_type][k]]), np.mean([x[1] for x in self.predicted_z_scores[model_type][k]]))
                print('correlation between GT and pred: ', scipy.stats.spearmanr([x[0] for x in self.predicted_z_scores[model_type][k]], 
                                                                            [x[1] for x in self.predicted_z_scores[model_type][k]]).correlation)
                corr_per_res[model_type][k] = [len(self.predicted_z_scores[model_type][k]), scipy.stats.spearmanr([x[0] for x in self.predicted_z_scores[model_type][k]], 
                                                                            [x[1] for x in self.predicted_z_scores[model_type][k]]).correlation]
                print()
        return corr_per_res

    def get_z_scores_per_model(self):
        actual_z_scores = []
        z_scores_per_model = {model_type: [] for model_type in self.predicted_z_scores.keys()}
        for model_type in z_scores_per_model.keys():
            for key in self.predicted_z_scores[model_type].keys():
                for i in range(len(self.predicted_z_scores[model_type][key])):
                    z_scores_per_model[model_type].append(self.predicted_z_scores[model_type][key][i][1])
                    if model_type == 'odin':
                        actual_z_scores.append(self.predicted_z_scores[model_type][key][i][0]) 
        return actual_z_scores, z_scores_per_model


def main(args):
    if args.train_strategy not in constants.train_strategies:
        print("The training strategies are:")
        print(*constants.train_strategies, sep="\n")
        sys.exit(2)

    if (args.model_type not in constants.model_types) and (
        args.model_type != "combined"
    ):
        print("The pre-trained models are:")
        print(*constants.model_types, sep="\n")
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
    predicted_z_scores = main(args.benchmark_data_path,
                              args.train_json_file,
                              args.test_json_file,
                              args.train_repr_dir,
                              args.test_repr_dir,
                              args.train_strategy)
    chezod_compare = CheZodCompare(predicted_z_scores)
    corr_per_res = chezod_compare.get_corr_per_residue()
    actual_z_scores, z_scores_per_model = chezod_compare.get_z_scores_per_model()
    plot_corr_per_residue(corr_per_res, 'esm-1b')
    plot_gt_vs_pred_contours(actual_z_scores, z_scores_per_model)
    