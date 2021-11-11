# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pandas as pd
from sklearn.model_selection import KFold
from sklearn import linear_model
import scipy 
import numpy as np
from joblib import dump
import sys
import getopt
import utils
import constants
from adopt import CheZod

# disorder predictor training

class DisorderPred:
    def __init__(self, 
                 path_chezod_1325_raw, 
                 path_chezod_117_raw,
                 path_chezod_1325_repr,
                 path_chezod_117_repr):
        self.path_chezod_1325_raw = str(path_chezod_1325_raw)
        self.path_chezod_117_raw = str(path_chezod_117_raw)
        self.path_chezod_1325_repr = str(path_chezod_1325_repr)
        self.path_chezod_117_repr = str(path_chezod_117_repr)
        chezod = CheZod(self.path_chezod_1325_raw, self.path_chezod_117_raw)
        self.ex_train, self.zed_train, self.ex_test, self.zed_test = chezod.get_train_test_sets(self.path_chezod_1325_repr, 
                                                                                                self.path_chezod_117_repr)
        _, self.df_ch, _ = chezod.get_chezod_raw()
        self.repr_path = utils.representation_path(self.path_chezod_1325_repr,
                                                   self.path_chezod_117_repr)

    def cleared_residue(self):
        # residue level split, train on cleared chezod 1325 and validation on chezod 117
        CorrelationsLR = {}
        LinearRegressions = {}

        for model_type in constants.model_types:
            reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
            reg_.fit(self.ex_train[model_type], self.zed_train[model_type])
            
            print(f'{model_type} - Correlation between the predicted and the ground trouth on the test set: ', 
                scipy.stats.spearmanr(self.zed_test[model_type], reg_.predict(self.ex_test[model_type])).correlation)
            
            LinearRegressions[model_type] = reg_
            CorrelationsLR[model_type] = scipy.stats.spearmanr(self.zed_test[model_type], reg_.predict(self.ex_test[model_type])).correlation
            dump(reg_, '../models/lasso_'+model_type+'_cleared_residue.joblib') 
        
        # ESM-1v and ESM-1b Combined 
        # --------------------------
        ex_train_combined = np.concatenate((self.ex_train['esm-1v'], self.ex_train['esm-1b']), axis=1)
        ex_test_combined = np.concatenate((self.ex_test['esm-1v'], self.ex_test['esm-1b']), axis=1)

        reg = linear_model.Lasso(alpha=0.0001, max_iter=10000)
        reg.fit(ex_train_combined, self.zed_train['esm-1v'])
            
        print('Combining esm-1v and esm-1b - Correlation between the predicted and the ground trouth on the test set: ', 
                scipy.stats.spearmanr(self.zed_test['esm-1v'], reg.predict(ex_test_combined)).correlation)

        # save the combined regression
        LinearRegressions['combined'] = reg
        dump(reg, '../models/lasso_combined_cleared_residue.joblib') 


    def residue_cv(self):
        # assemble the training data from the 1325 set
        #ex_1325, zed_1325 = pedestrian_input(list(df_ch['brmid']), df_ch, path_chezod_esm_repr, z_col='z-score')
        # read the data 
        ex_1325, zed_1325 = {}, {}

        for model_type in constants.model_types:
            if model_type=='esm-msa':
                msa_ind=True
            else:
                msa_ind=False
            
            # assemble the training data from the 1325 set
            ex_1325[model_type], zed_1325[model_type] = utils.pedestrian_input(list(self.df_ch['brmid']), self.df_ch, self.repr_path[model_type]['1325'], z_col='z-score', msa=msa_ind)
        
        # 10 fold CV on the 1325 set 
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(ex_1325[constants.model_types[0]]) # since the number of inputs are the same for all 3 models, 
                                                           # and the splits are based on indices, it's okay to select indices based only 
                                                           # on one of the model types
        corrs = {model_type: [] for model_type in constants.model_types}
        regressors = {model_type: [] for model_type in constants.model_types}
        rounds = 1

        for train_index, test_index in kf.split(ex_1325[constants.model_types[0]]):
            print('rounds: ', rounds)
            print('-----------------')
                
            for model_type in constants.model_types:
                print(model_type)
                ex_rounds_train = np.take(ex_1325[model_type], train_index, axis=0)
                ex_rounds_test = np.take(ex_1325[model_type], test_index, axis=0)
            
                zed_rounds_train = np.take(zed_1325[model_type], train_index, axis=0)
                zed_rounds_test = np.take(zed_1325[model_type], test_index, axis=0)

                reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
                reg_.fit(ex_rounds_train, zed_rounds_train)
            
                corrs[model_type].append(scipy.stats.spearmanr(zed_rounds_test, reg_.predict(ex_rounds_test)).correlation)
                regressors[model_type].append(reg_)
            
                print('Correlation between the predicted and the ground trouth on the test set: ', 
                    scipy.stats.spearmanr(zed_rounds_test, reg_.predict(ex_rounds_test)).correlation)
                print()
            rounds+=1

        for model_type in constants.model_types:
            # save best regressor for inference
            index_min_corr = min(range(len(corrs[model_type])), key=corrs[model_type].__getitem__)    
            best_reg = regressors[model_type][index_min_corr]
            dump(best_reg, '../models/lasso_'+model_type+'_residue_cv.joblib') 

            print(model_type)
            print('10-fold CV - average correlation between the predicted and the ground trouth on the test set: ', np.mean(corrs[model_type]))
            print()


    def cleared_residue_cv(self):
        # 10 fold CV on the cleared 1325 set, i.e. removing the proteins that appear also in the 117 set 
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(self.ex_train[constants.model_types[0]])

        corrs_cleared = {model_type: [] for model_type in constants.model_types}
        regressors = {model_type: [] for model_type in constants.model_types}
        rounds = 1

        for train_index, test_index in kf.split(self.ex_train[constants.model_types[0]]):
            print('rounds: ', rounds)
            print('-----------------')
                
            for model_type in constants.model_types:
                ex_rounds_train = np.take(self.ex_train[model_type], train_index, axis=0)
                ex_rounds_test = np.take(self.ex_train[model_type], test_index, axis=0)
            
                zed_rounds_train = np.take(self.zed_train[model_type], train_index, axis=0)
                zed_rounds_test = np.take(self.zed_train[model_type], test_index, axis=0)

                reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
                reg_.fit(ex_rounds_train, zed_rounds_train) 
            
                corrs_cleared[model_type].append(scipy.stats.spearmanr(zed_rounds_test, reg_.predict(ex_rounds_test)).correlation)
                regressors[model_type].append(reg_)
            
                print('Correlation between the predicted and the ground trouth on the test set: ', 
                    scipy.stats.spearmanr(zed_rounds_test, reg_.predict(ex_rounds_test)).correlation)
            
                print()
            rounds+=1

        for model_type in constants.model_types:
            # save best regressor for inference
            index_min_corr = min(range(len(corrs_cleared[model_type])), key=corrs_cleared[model_type].__getitem__)    
            best_reg = regressors[model_type][index_min_corr]
            dump(best_reg, '../models/lasso_'+model_type+'_cleared_residue_cv.joblib') 

            print(model_type)
            print('10-fold CV (on the reduced 1325, i.e. overlap removed) - ')
            print('average correlation between the predicted and the ground trouth on the test set: ', np.mean(corrs_cleared[model_type]))


    def cleared_sequence_cv(self):
        # 10-fold CV protein sequence based fold selection applied on cleared chezod 1325
        seq_ids = list(self.df_ch['brmid'])

        # 10 fold CV on the cleared 1325 set, i.e. removing the proteins that appear also in the 117 set 
        kf = KFold(n_splits=10, shuffle=True)
        kf.get_n_splits(seq_ids)

        corrs_seq = {model_type: [] for model_type in constants.model_types}
        regressors = {model_type: [] for model_type in constants.model_types}
        rounds_seq = 1

        for train_index, test_index in kf.split(seq_ids):
            print('rounds: ', rounds_seq)
            print('-----------------')
                
            for model_type in constants.model_types:
                
                if model_type=='esm-msa':
                    msa_ind=True
                else:
                    msa_ind=False
                
                # assemble the training data from the cleared 1325 set
                train_brmids = np.take(seq_ids, train_index, axis=0)
                test_brmids =np.take(seq_ids, test_index, axis=0)
            
                ex_train_seq, zed_train_seq = utils.pedestrian_input(train_brmids, self.df_ch, self.repr_path[model_type]['1325'], z_col='z-score', msa=msa_ind, drop_missing=True)
                ex_test_seq, zed_test_seq = utils.pedestrian_input(test_brmids, self.df_ch, self.repr_path[model_type]['1325'], z_col='z-score', msa=msa_ind, drop_missing=True)

                reg_ = linear_model.Lasso(alpha=0.0001, max_iter=10000)
                reg_.fit(ex_train_seq, zed_train_seq)
            
                corrs_seq[model_type].append(scipy.stats.spearmanr(zed_test_seq, reg_.predict(ex_test_seq)).correlation)
                regressors[model_type].append(reg_)
            
                print('Correlation between the predicted and the ground trouth on the test set: ', 
                    scipy.stats.spearmanr(zed_test_seq, reg_.predict(ex_test_seq)).correlation)
            
                print()
            rounds_seq+=1

        for model_type in constants.model_types:
            # save best regressor for inference
            index_min_corr = min(range(len(corrs_seq[model_type])), key=corrs_seq[model_type].__getitem__)    
            best_reg = regressors[model_type][index_min_corr]
            dump(best_reg, '../models/lasso_'+model_type+'_cleared_sequence_cv.joblib') 

            print(model_type)
            print('10-fold CV - Folds split on sequence level')
            print('average correlation between the predicted and the ground trouth on the test set: ', np.mean(corrs_seq[model_type]))


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hs:t:e:r:p:", ["train_strategy=", "train_json_file=", "test_json_file=", "train_repr_dir=", "test_repr_dir="]) 
    except getopt.GetoptError:
        print('usage: training.py -s <training_strategy> -t <train_json_file_path> -e <test_json_file_path=> -r <train_residue_level_representation_dir> -p <test_residue_level_representation_dir>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: training.py -s <training_strategy> -t <train_json_file_path=> -e <test_json_file_path=> -r <train_residue_level_representation_dir> -p <test_residue_level_representation_dir>')
            sys.exit()
        elif opt in ("-s", "--train_strategy"):
            train_strategy = arg
            if train_strategy not in constants.train_strategies:
                print("The training strategies are:")
                print(*constants.train_strategies, sep="\n")
                sys.exit(2)
        elif opt in ("-t", "--train_json_file"):
            train_sequences = arg
        elif opt in ("-e", "--test_json_file"):
            test_sequences = arg
        elif opt in ("-r", "--train_repr_dir"):
            train_repr_dir = arg
        elif opt in ("-p", "--test_repr_dir"):
            test_repr_dir = arg

    disorder_pred = DisorderPred(train_sequences, 
                                    test_sequences,
                                    train_repr_dir,
                                    test_repr_dir)

    if train_strategy == "train_on_cleared_1325_test_on_117_residue_split":
        disorder_pred.cleared_residue()
    elif train_strategy == "train_on_1325_cv_residue_split":
        disorder_pred.residue_cv()
    elif train_strategy == "train_on_cleared_1325_cv_residue_split":
        disorder_pred.cleared_residue_cv()
    else:
        disorder_pred.cleared_sequence_cv()

if __name__ == "__main__":
    main(sys.argv[1:])

