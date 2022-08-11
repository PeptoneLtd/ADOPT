# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch

from adopt import constants, utils


class MultiHead:
    def __init__(self, model_type):
        self.model_type = model_type
        self.models, self.alphabets, self.msa = utils.get_model_alphabet_msa(model_type)


    def get_results(self, sequence, brmid):
        results = []
        i=0
        while i < len(self.models):
            results.append(utils.get_esm_output(self.models[i], self.alphabets[i], [(brmid, sequence)], self.msa[i]))
            i+=1
        return results


    def get_attention(self, sequence, brmid):
        if self.model_type in constants.msa_model_types:
            results = self.get_results(sequence, brmid)
        else:
            print("The model types are:")
            print(*self.model_types, sep="\n")
            sys.exit(2)

        tokens = list(sequence)
        if self.model_type == 'esm-msa':
            attention = results["row_attentions"].permute(1, 0, 2, 3, 4)
        else:
            attention = results["attentions"].permute(1, 0, 2, 3, 4)
        # remove first and last token (<cls> and <sep>)
        attention = attention[:, :, :, 1:-1, 1:-1]
        return attention, tokens

    def get_representation(self, sequence, brmid):
        results = self.get_results(sequence, brmid)
        
        if self.model_type in constants.model_types:
            representation = results[0]["representations"][33]
        elif self.model_type == 'esm-msa':
            representation = results[0]["representations"][12][0]
        elif self.model_type == "combined":
            representation = torch.cat((results[0]["representations"][33], results[1]["representations"][33]), -1)
        else:
            print("The model types are:")
            print(*constants.msa_model_types, sep="\n")
            sys.exit(2)

        tokens = list(sequence)
        # remove first and last token (<cls> and <sep>)
        representation = representation[:, 1:-1, :]
        return representation, tokens
