# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

esm_models = ["esm1v_t33_650M_UR90S_1", "esm1b_t33_650M_UR50S"]
# "esm_msa1b_t12_100M_UR50S"]

model_types = ["esm-1v", "esm-1b"]  # , 'esm-msa']

models_dict = {"esm1v_t33_650M_UR90S_1": "esm-1v", "esm1b_t33_650M_UR50S": "esm-1b"}
# "esm_msa1b_t12_100M_UR50S":'esm-msa'}

train_strategies = [
    "train_on_cleared_1325_test_on_117_residue_split",
    "train_on_1325_cv_residue_split",
    "train_on_cleared_1325_cv_residue_split",
    "train_on_cleared_1325_cv_sequence_split",
]

strategies_dict = {
    "train_on_cleared_1325_test_on_117_residue_split": "cleared_residue",
    "train_on_1325_cv_residue_split": "residue_cv",
    "train_on_cleared_1325_cv_residue_split": "cleared_residue_cv",
    "train_on_cleared_1325_cv_sequence_split": "cleared_sequence_cv",
}
