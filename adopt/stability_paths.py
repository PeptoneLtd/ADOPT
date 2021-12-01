# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import scipy
from joblib import Parallel, delayed
from sklearn import linear_model

from adopt import constants, CheZod, utils

class StabilityAnalysis:
    def __init__(self,
                 path_chezod_1325_raw,
                 path_chezod_117_raw,
                 path_chezod_1325_repr,
                 path_chezod_117_repr):
        self.path_chezod_1325_raw = path_chezod_1325_raw
        self.path_chezod_117_raw = path_chezod_117_raw
        self.path_chezod_1325_repr = path_chezod_1325_repr
        self.path_chezod_117_repr = path_chezod_117_repr

    def get_stability_paths(self, model_picked):
            chezod = CheZod(self.path_chezod_1325_raw, self.path_chezod_117_raw)
            (
                ex_train,
                zed_train,
                _,
                _,
            ) = chezod.get_train_test_sets(self.path_chezod_1325_repr, self.path_chezod_117_repr)
            
            nr_samples = constants.stability_path_reg_params['nr_samples']
            reg_params = np.linspace(constants.stability_path_reg_params['start'], 
                                    constants.stability_path_reg_params['end'], 
                                    constants.stability_path_reg_params['n_points'])
            sample_size = ex_train[model_picked].shape[0]//2
            # collect the probabilities of being selected
            probabs = {} 
            for reg_param in reg_params:
                print("Computing stability path for regularisation parameter: ", reg_param)
                print('------------------')
                selected_idxs = np.zeros(ex_train[model_picked].shape[1])
                abs_reg_coefs = Parallel(n_jobs=-1)(delayed(utils.stability_selection_prob)(ex_train, 
                                                                                            zed_train, 
                                                                                            model_picked, 
                                                                                            sample_size, 
                                                                                            reg_param) for _ in range(nr_samples))
                for reg_coef in abs_reg_coefs:
                    np.add.at(selected_idxs, np.where(reg_coef != 0.0)[0], 1)
                probabs[reg_param] = 1./nr_samples*selected_idxs
            return probabs
        
    def plot_stability_paths(self, probabs, model_picked):
        cutoffs = constants.stability_path_hyperparams['cutoffs']
        freq_cutoff = constants.stability_path_hyperparams['freq_cutoff']
        chezod = CheZod(self.path_chezod_1325_raw, self.path_chezod_117_raw)
        (
            ex_train,
            zed_train,
            ex_test,
            zed_test,
        ) = chezod.get_train_test_sets(self.path_chezod_1325_repr, self.path_chezod_117_repr)
        for cutoff in cutoffs:
            fig = go.Figure()
            coordinates_relevant = []
            for i in range(ex_train[model_picked].shape[1]):
                if sum([1. if probabs[key][i] > cutoff else 0. for key in probabs.keys()])>freq_cutoff:
                    coordinates_relevant.append(i)
                    line = dict(color="red", width=.5)
                else:
                    line=dict(dash='dash', color="black", width=.5)
                fig.add_trace(go.Scatter(x=list(probabs.keys()), 
                                    y=[probabs[key][i] for key in probabs.keys()],
                                    mode='lines',
                                    line=line))
            fig.update_layout(
                #title=f"Cutoff {cutoff}",
                xaxis_title=r"$\lambda$",
                yaxis_title=r"$\Pi(\lambda)$",
                legend_title="Legend Title",
                font=dict(
                    family="Courier New",
                    size=18,
                    color="black"
                )
            )
            fig.update_layout(showlegend=False)
            pio.write_image(fig, 
                            "../media/stability_paths_" 
                            + "_cp_"
                            + str(cutoff)
                            + "_cf_"
                            + str(freq_cutoff)
                            + ".pdf", 
                            width=900, 
                            height=450, 
                            scale=1.) #, width=800, height=800, scale=1.0)
            # do the Lasso
            ex_train_filtered = np.take(ex_train[model_picked], coordinates_relevant, axis=1)
            ex_test_filtered = np.take(ex_test[model_picked], coordinates_relevant, axis=1)
            # slim regression
            reg_slim_lasso = linear_model.Lasso(alpha=0.0001, max_iter=10000)
            reg_slim_lasso.fit(ex_train_filtered, zed_train[model_picked])  
            print('Stability selection with cp=', cutoff, "and cf=", freq_cutoff)
            print('Coordinate ', len(coordinates_relevant), 'correlation: ', scipy.stats.spearmanr(zed_test[model_picked], reg_slim_lasso.predict(ex_test_filtered)).correlation)