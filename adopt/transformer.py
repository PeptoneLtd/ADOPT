# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys

import torch

from adopt import constants, utils


class MultiHead:
    def __init__(self, model_type, sequence, brmid):
        self.model_type = model_type
        self.sequence = sequence
        self.brmid = brmid
        self.data = [(self.brmid, self.sequence)]

    def get_attention(self):
        if self.model_type in constants.model_types:
            results = utils.get_model_and_alphabet(self.model_type, self.data)
        else:
            print("The model types are:")
            print(*constants.model_types, sep="\n")
            sys.exit(2)

        tokens = list(self.sequence)
        attention = results["attentions"].permute(1, 0, 2, 3, 4)
        # remove first and last token (<cls> and <sep>)
        attention = attention[:, :, :, 1:-1, 1:-1]
        return attention, tokens

    def get_representation(self):
        if self.model_type in constants.model_types:
            results = utils.get_model_and_alphabet(self.model_type, self.data)
            representation = results["representations"][33]
        elif self.model_type == "combined":
            results_esm1b, results_esm1v = utils.get_model_and_alphabet(
                self.model_type, self.data
            )
            representation_esm1b = results_esm1b["representations"][33]
            representation_esm1v = results_esm1v["representations"][33]
            representation = torch.cat((representation_esm1b, representation_esm1v), -1)
        else:
            print("The model types are:")
            print(*constants.model_types, sep="\n")
            sys.exit(2)

        tokens = list(self.sequence)
        # remove first and last token (<cls> and <sep>)
        representation = representation[:, 1:-1, :]
        return representation, tokens
