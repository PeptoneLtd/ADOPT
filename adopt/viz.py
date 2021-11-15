import esm
import torch
from bertviz import head_view, model_view

from adopt import constants


def viz_attention(model_type, sequence, brmid):
    # Load ESM model
    if model_type == "esm-1b":
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
    elif model_type == "esm-1v":
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    # elif model_type == 'esm-msa':
    #    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S
    else:
        print("The model types are:")
        print(*constants.model_types, sep="\n")
    batch_converter = alphabet.get_batch_converter()

    data = [(brmid, sequence)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    tokens = list(sequence)
    attention = results["attentions"].permute(1, 0, 2, 3, 4)
    # remove first and last token (<cls> and <sep>)
    attention = attention[:, :, :, 1:-1, 1:-1]
    return attention, tokens
