import esm
import torch

from adopt import constants, utils


def get_multi_attention(model_type, sequence, brmid):
    # Load ESM model
    if model_type == "esm-1b":
        model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        results = utils.get_esm_attention(model, alphabet, sequence, brmid)
    elif model_type == "esm-1v":
        model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
        results = utils.get_esm_attention(model, alphabet, sequence, brmid)
    elif model_type == 'combined':
        model_esm1b, alphabet_esm1b = esm.pretrained.esm1b_t33_650M_UR50S()
        model_esm1v, alphabet_esm1v = esm.pretrained.esm1v_t33_650M_UR90S_1()
        results_esm1b = utils.get_esm_attention(model_esm1b, alphabet_esm1b, sequence, brmid)
        results_esm1v = utils.get_esm_attention(model_esm1v, alphabet_esm1v, sequence, brmid)
        results = torch.cat((results_esm1b, results_esm1v), 0)
    # elif model_type == 'esm-msa':
    #    model, alphabet = esm.pretrained.esm_msa1b_t12_100M_UR50S
    else:
        print("The model types are:")
        print(*constants.model_types, sep="\n")

    tokens = list(sequence)

    if model_type == 'combined':
        attention_esm1b = results_esm1b["attentions"].permute(1, 0, 2, 3, 4)
        attention_esm1v = results_esm1v["attentions"].permute(1, 0, 2, 3, 4)
        attention = torch.cat((attention_esm1b, attention_esm1v), 1)
    else:
        attention = results["attentions"].permute(1, 0, 2, 3, 4)
    # remove first and last token (<cls> and <sep>)
    attention = attention[:, :, :, 1:-1, 1:-1]
    return attention, tokens
