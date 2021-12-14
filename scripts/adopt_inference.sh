#!/bin/bash  

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

NEW_PROT_FASTA_FILE_PATH="<new_proteins_fasta_file_path>" # put here your path
NEW_PROT_RES_REPR_DIR_PATH="<new_proteins_residue_level_representation_dir>" # put here your path
TRAIN_STRATEGY="train_on_cleared_1325_test_on_117_residue_split"
MODEL_TYPE="esm-1b"
PRED_Z_FILE_PATH="predicted_z_scores.json"

printf "Extracting residue level representation of %s \n" $NEW_PROT_FASTA_FILE_PATH

python adopt/embedding.py $NEW_PROT_FASTA_FILE_PATH \
                          $NEW_PROT_RES_REPR_DIR_PATH

printf "Computing Z scores of %s \n" $NEW_PROT_FASTA_FILE_PATH

python adopt/inference.py $NEW_PROT_FASTA_FILE_PATH \
                          $NEW_PROT_RES_REPR_DIR_PATH \
                          $PRED_Z_FILE_PATH \
                          --train_strategy $TRAIN_STRATEGY \
                          --model_type $MODEL_TYPE


