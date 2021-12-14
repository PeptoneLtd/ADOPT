#!/bin/bash  

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The work directory is </ADOPT> in the <adopt> container
LOCAL_MSA_DIR="<uniclast30_and_msas_dir>" # put here your path 
NEW_PROT_FASTA_FILE_PATH="<new_proteins_fasta_file_path>" # put here your path
NEW_PROT_RES_REPR_DIR_PATH="<new_proteins_residue_level_representation_dir>" # put here your path
TRAIN_STRATEGY="train_on_cleared_1325_test_on_117_residue_split"
MODEL_TYPE="esm-msa"
PRED_Z_FILE_PATH="predicted_z_scores.json"

printf "Setting up the MSA procedure \n"
bash scripts/adopt_msa_setup.sh $LOCAL_MSA_DIR

printf "Extracting Multi Sequence Alignments of %s \n" $NEW_PROT_FASTA_FILE_PATH
bash scripts/msa_generator.sh $NEW_PROT_FASTA_FILE_PATH

printf "Extracting residue level representation of %s \n" $NEW_PROT_FASTA_FILE_PATH
docker exec -it adopt python adopt/embedding.py $NEW_PROT_FASTA_FILE_PATH \
                                                $NEW_PROT_RES_REPR_DIR_PATH \
                                                --msa

printf "Computing Z scores of %s \n" $NEW_PROT_FASTA_FILE_PATH
docker exec -it adopt python adopt/inference.py $NEW_PROT_FASTA_FILE_PATH \
                                                $NEW_PROT_RES_REPR_DIR_PATH \
                                                $PRED_Z_FILE_PATH \
                                                --train_strategy $TRAIN_STRATEGY \
                                                --model_type $MODEL_TYPE 

docker cp adopt:/ADOPT/$PRED_Z_FILE_PATH $PRED_Z_FILE_PATH
printf "The predicted Z scores have been stored in %s" $PRED_Z_FILE_PATH



