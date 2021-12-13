#!/bin/bash 

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# The work directory is </ADOPT> in the <adopt> container
LOCAL_MSA_DIR="<uniclast30_and_msas_dir>" # put here your path
TRAIN_FASTA_FILE_PATH="datasets/chezod_1325_all.fasta"
TEST_FASTA_FILE_PATH="datasets/chezod_117_all.fasta"
TRAIN_RES_REPR_DIR_PATH="representations/chezod_1325"
TEST_RES_REPR_DIR_PATH="representations/chezod_117"
TRAIN_STRATEGY="<training_strategy>" # choose a training strategy
TRAIN_JSON_FILE_PATH="datasets/1325_dataset_raw.json"
TEST_JSON_FILE_PATH="datasets/117_dataset_raw.json" 

printf "Setting up the MSA procedure \n"
bash scripts/adopt_msa_setup.sh $LOCAL_MSA_DIR

printf "Extracting Multi Sequence Alignments of %s \n" $TRAIN_FASTA_FILE_PATH
bash scripts/msa_generator.sh $TRAIN_FASTA_FILE_PATH

printf "Extracting Multi Sequence Alignments of %s \n" $TEST_FASTA_FILE_PATH
bash scripts/msa_generator.sh $TEST_FASTA_FILE_PATH

printf "Extracting residue level representations of %s \n" $TRAIN_FASTA_FILE_PATH
docker exec -it adopt python adopt/embedding.py $TRAIN_FASTA_FILE_PATH \
                                                $TRAIN_RES_REPR_DIR_PATH \
                                                --msa

printf "Extracting residue level representations of %s \n" $TEST_FASTA_FILE_PATH
docker exec -it adopt python adopt/embedding.py $TEST_FASTA_FILE_PATH \
                                                $TEST_RES_REPR_DIR_PATH \
                                                --msa

printf "Training ADOPT on %s \n" $TRAIN_FASTA_FILE_PATH
docker exec -it adopt python adopt/training.py $TRAIN_JSON_FILE_PATH \
                                               $TEST_JSON_FILE_PATH \
                                               $TRAIN_RES_REPR_DIR_PATH \
                                               $TEST_RES_REPR_DIR_PATH \
                                               --train_strategy $TRAIN_STRATEGY \
                                               --msa

docker cp adopt:/ADOPT/models .
printf "The trained models have been stored in ./models"