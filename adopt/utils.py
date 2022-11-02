# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import esm
import numpy as np
import onnxruntime as rt
import pandas as pd
import torch
from Bio import SeqIO
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn import linear_model
import os

from adopt import constants

"@generated"
# throw away the missing values, if the drop_missing flag is set to True, i.e. where z-scores are  999
def pedestrian_input(indexes, df, path, z_col="z-score", msa=False, drop_missing=True):
    zeds = []
    exes = []

    for k in range(len(indexes)):
        if not msa:
            repr_esm = (
                torch.load(f"{path}{indexes[k]}.pt")["representations"][33]
                .clone()
                .cpu()
                .detach()
            )
        else:
            repr_esm = (
                torch.load(f"{path}{indexes[k]}.pt")["representations"]
                .clone()
                .cpu()
                .detach()
            )
        z_s = np.array(df[df["brmid"] == indexes[k]][z_col].to_numpy()[0])
        if drop_missing:
            idxs = np.where(z_s != 999)[0]
        else:
            idxs = np.arange(len(z_s))

        for i in idxs:
            zeds.append(z_s[i])
            exes.append(repr_esm[i].numpy())
    return np.array(exes), np.array(zeds)


# collect the path to representations according to model type and train vs test set
def representation_path(path_chezod_1325_repr, path_chezod_117_repr, msa):
    repr_path = {}
    if msa:
        model_types = constants.msa_model_types
    else:
        model_types = constants.model_types
    for model_type in model_types:
        repr_path[model_type] = {
            "1325": str(path_chezod_1325_repr) + "/" + model_type + "/",
            "117": str(path_chezod_117_repr) + "/" + model_type + "/",
        }
    return repr_path


def df_to_fasta(df, fasta_out_path):
    ofile = open(fasta_out_path, "w")
    for index, row in df.iterrows():
        ofile.write(">" + row["brmid"] + "\n" + row["sequence"] + "\n")
    ofile.close()


def fasta_to_df(fasta_input):
    with open(fasta_input) as fasta_file:
        identifiers = []
        sequences = []
        for seq_record in SeqIO.parse(fasta_file, "fasta"):  # (generator)
            identifiers.append(seq_record.id)
            sequences.append("".join(seq_record.seq))

    df = pd.DataFrame({"brmid": identifiers, "sequence": sequences})
    return df


def save_onnx_model(columns_shape, reg, model_name):
    initial_type = [("float_input", FloatTensorType([None, columns_shape]))]
    onx = convert_sklearn(reg, initial_types=initial_type)
    with open(model_name, "wb") as f:
        f.write(onx.SerializeToString())


def get_onnx_model_preds(model_name, input_data):
    sess = rt.InferenceSession(model_name)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred_onx = sess.run([label_name], {input_name: input_data})[0]
    return pred_onx


def get_esm_output(model, alphabet, data, msa):
    if msa:
        msa_transformer = model.eval()
        msa_batch_converter = alphabet.get_batch_converter()
        _, _, msa_batch_tokens = msa_batch_converter(data)
        with torch.no_grad():
            results = msa_transformer(msa_batch_tokens, repr_layers=[12], return_contacts=True)
    else:
        batch_converter = alphabet.get_batch_converter()
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    return results


def load_model_and_alphabet(model_name):
    models_dir = os.environ.get("ESM_MODELS_DIR")
    try: # try to find model locally
        path = os.path.join(models_dir, f"{model_name}.pt")
        print(f"Trying to load ESM model from {path}")
        return esm.pretrained.load_model_and_alphabet_local(path)
    #TODO: catch appropriate exception 
    except Exception as e:
        print(type(e))    # the exception instance
        print(e.args)     # arguments stored in .args
        print(e)
        print("ESM model not found locally, downloading from torchub")
        return esm.pretrained.load_model_and_alphabet_hub(model_name)


def get_model_and_alphabet(model_type, data):
    # Load ESM model
    if model_type == "esm-1b":
        model, alphabet = load_model_and_alphabet("esm1b_t33_650M_UR50S")
        results = get_esm_output(model, alphabet, data, False)
    elif model_type == "esm-1v":
        model, alphabet = load_model_and_alphabet("esm1v_t33_650M_UR90S_1")
        results = get_esm_output(model, alphabet, data, False)
    elif model_type == 'esm-msa':
        model, alphabet = load_model_and_alphabet("esm_msa1b_t12_100M_UR50S")
        results = get_esm_output(model, alphabet, data, True)
    else:
        model_esm1b, alphabet_esm1b = load_model_and_alphabet("esm1b_t33_650M_UR50S")
        model_esm1v, alphabet_esm1v = load_model_and_alphabet("esm1v_t33_650M_UR90S_1")
        results_esm1b = get_esm_output(model_esm1b, alphabet_esm1b, data)
        results_esm1v = get_esm_output(model_esm1v, alphabet_esm1v, data)
        results = [results_esm1b, results_esm1v]
    return results


def get_model_alphabet_msa(model_type):
    models = []
    alphabets = []
    msa = []

    if model_type == "esm-1b":
        model, alphabet = load_model_and_alphabet("esm1b_t33_650M_UR50S")
        return [model], [alphabet], [False]
    
    if model_type == "esm-1v":
        model, alphabet = load_model_and_alphabet("esm1v_t33_650M_UR90S_1")
        return [model], [alphabet], [False]
    
    if model_type == 'esm-msa':
        model, alphabet = load_model_and_alphabet("esm_msa1b_t12_100M_UR50S")
        return [model], [alphabet], [True]
    
    for s in ("esm1b_t33_650M_UR50S", "esm1v_t33_650M_UR90S_1"):
        m, a = load_model_and_alphabet(s)
        models.append(m)
        alphabets.append(a)
        msa.append(False)
        
    return models, alphabets, msa
    

def get_residue_class(predicted_z_scores):
    residues_state = []
    for n, zscore in enumerate(predicted_z_scores):
        residues_dict = {}
        if zscore < 3:
            residues_dict["label"] = constants.structure_dict["Fully disordered"]
        elif 3 <= zscore < 8:
            residues_dict["label"] = constants.structure_dict["Partially disordered"]
        elif zscore >= 11:
            residues_dict["label"] = constants.structure_dict["Structured"]
        else:
            residues_dict["label"] = constants.structure_dict["Flexible loops"]

        residues_dict["start"] = n
        residues_dict["end"] = n + 1
        residues_state.append(residues_dict)
    return residues_state


def stability_selection_prob(ex_train, zed_train, model_picked, sample_size, reg_param):
    sample_idxs = np.random.choice(np.arange(ex_train[model_picked].shape[0]), sample_size, replace=False) 
    ex_train_filtered = np.take(ex_train[model_picked], sample_idxs, axis=0)
    zed_train_filtered = np.take(zed_train[model_picked], sample_idxs, axis=0)
    # Lasso regression
    reg = linear_model.Lasso(alpha=reg_param, max_iter=10000)
    reg.fit(ex_train_filtered, zed_train_filtered)
    reg_coef = abs(reg.coef_)
    return reg_coef

  
def get_esm_models(msa):
    if msa:
        models = constants.esm_msa_models
    else:
        models = constants.esm_models
    return models


def get_model_types(msa):
    if msa:
        model_types = constants.msa_model_types
    else:
        model_types = constants.model_types
    return model_types
