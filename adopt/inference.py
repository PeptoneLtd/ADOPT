# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from adopt import utils
from joblib import load
import torch
import os
import re
import getopt
import sys
import constants

def get_z_score(strategy,
                model_type,
                inference_fasta_path,
                inference_repr_path,
                predicted_z_scores_path):
    df_fasta = utils.fasta_to_df(inference_fasta_path)

    if model_type == "combined":
        repr_path = inference_repr_path+"/"+'esm-1v'
    else:
        repr_path = inference_repr_path+"/"+model_type

    repr_files = os.listdir(repr_path)
    indexes = []

    for file in repr_files:
        indexes.append(file.split('.')[0])

    reg = load('../models/lasso_'+model_type+'_'+constants.strategies_dict[strategy]+'.joblib')
    predicted_z_scores = []

    for ix in indexes:
        if model_type == "esm-msa":
            repr_esm = torch.load(str(repr_path)+"/"+ix+".pt")['representations'][12].clone().cpu().detach()
        elif model_type == "combined":
            esm1b_repr_path = inference_repr_path+"/"+'esm-1b'
            repr_esm1v = torch.load(str(repr_path)+"/"+ix+".pt")['representations'][33].clone().cpu().detach()
            repr_esm1b = torch.load(str(esm1b_repr_path)+"/"+ix+".pt")['representations'][33].clone().cpu().detach()
            repr_esm = torch.cat([repr_esm1v,repr_esm1b], 1)
        else:
            repr_esm = torch.load(str(repr_path)+"/"+ix+".pt")['representations'][33].clone().cpu().detach()
        z_scores = reg.predict(repr_esm)
        predicted_z_scores.append(z_scores)
        
    df_fasta['z_scores'] = predicted_z_scores
    df_fasta.to_json(predicted_z_scores_path, orient="records")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "hs:m:f:r:p:", ["train_strategy=", "model_type=", "infer_fasta_file=", "infer_repr_dir=", "pred_z_scores_file"]) 
    except getopt.GetoptError:
        print('usage: inference.py -s <training_strategy> -m <model_type> -f <inference_fasta_file> -r <inference_repr_dir> -p <predicted_z_scores_file>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('usage: inference.py -s <training_strategy> -m <model_type> -f <inference_fasta_file> -r <inference_repr_dir> -p <predicted_z_scores_file>')
            sys.exit()
        elif opt in ("-s", "--train_strategy"):
            train_strategy = arg
            if train_strategy not in constants.train_strategies:
                print("The training strategies are:")
                print(*constants.train_strategies, sep="\n")
                sys.exit(2)
        elif opt in ("-m", "--model_type"):
            model_type = arg
            if (model_type not in constants.model_types) and (model_type != "combined"):
                print("The pre-trained models are:")
                print(*constants.model_types, sep="\n")
                print("combined")
                sys.exit(2)
            if (train_strategy != "train_on_cleared_1325_test_on_117_residue_split") and (model_type=='combined'):
                print("Only the train_on_cleared_1325_test_on_117_residue_split strategy is allowed with the <combined> model")
                sys.exit()
        elif opt in ("-f", "--infer_fasta_file"):
            infer_fasta_file = arg
        elif opt in ("-r", "--infer_repr_dir"):
            infer_repr_dir = arg
        elif opt in ("-p", "--pred_z_scores_file"):
            pred_z_scores_file = arg

    get_z_score(train_strategy,
                model_type,
                infer_fasta_file,
                infer_repr_dir,
                pred_z_scores_file)

if __name__ == "__main__":
    main(sys.argv[1:])