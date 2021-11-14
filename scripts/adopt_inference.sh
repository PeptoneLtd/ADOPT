# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash   

NEW_PROT_FASTA_FILE_PATH="<new_proteins_fasta_file_path>"
NEW_PROT_RES_REPR_DIR_PATH="<new_proteins_residue_level_representation_dir>"
TRAIN_STRATEGY="<training_strategy>"
MODEL_TYPE="<model_type>"
PRED_Z_FILE_PATH="<predicted_z_scores_file>"

printf "Extracting residue level representation of $NEW_PROT_FASTA_FILE_PATH \n"

python ../adopt/embedding.py -f $NEW_PROT_FASTA_FILE_PATH \
                             -r $NEW_PROT_RES_REPR_DIR_PATH

printf "Computing Z scores of $NEW_PROT_FASTA_FILE_PATH \n"

python ../adopt/inference.py -s $TRAIN_STRATEGY \
                             -m $MODEL_TYPE \
                             -f $NEW_PROT_FASTA_FILE_PATH \
                             -r $NEW_PROT_RES_REPR_DIR_PATH \
                             -p $PRED_Z_FILE_PATH
