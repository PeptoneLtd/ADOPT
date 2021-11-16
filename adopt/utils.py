# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import onnxruntime as rt
import pandas as pd
import torch
from Bio import SeqIO
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from adopt import constants


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
def representation_path(path_chezod_1325_repr, path_chezod_117_repr):
    repr_path = {}
    for model_type in constants.model_types:
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


def get_esm_attention(model, alphabet, data):
    batch_converter = alphabet.get_batch_converter()
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
    return results
