#!/bin/bash 

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

LOCAL_MSA_DIR="../msa"
TRAIN_FASTA_FILE_PATH="../datasets/chezod_1325_all.fasta"
TEST_FASTA_FILE_PATH="../datasets/chezod_117_all.fasta"
TRAIN_RES_REPR_DIR_PATH="../representations/chezod_1325"
TEST_RES_REPR_DIR_PATH="../representations/chezod_117"
TRAIN_STRATEGY="<training_strategy>" # choose a training strategy
TRAIN_JSON_FILE_PATH="../datasets/1325_dataset_raw.json"
TEST_JSON_FILE_PATH="../datasets/117_dataset_raw.json" 

printf "Setting up the MSA procedure \n"
bash adopt_msa_setup.sh $LOCAL_MSA_DIR

printf "Extracting Multi Sequence Alignments of %s \n" $TRAIN_FASTA_FILE_PATH
bash msa_generator.sh $TRAIN_FASTA_FILE_PATH

printf "Extracting Multi Sequence Alignments of %s \n" $TEST_FASTA_FILE_PATH
bash msa_generator.sh $TEST_FASTA_FILE_PATH

printf "Extracting residue level representations of %s \n" $TRAIN_FASTA_FILE_PATH
docker exec -it adopt bash python ../adopt/embedding.py -f $TRAIN_FASTA_FILE_PATH \
                                                        -r $TRAIN_RES_REPR_DIR_PATH \
                                                        -m

printf "Extracting residue level representations of %s \n" $TEST_FASTA_FILE_PATH
docker exec -it adopt python ../adopt/embedding.py -f $TEST_FASTA_FILE_PATH \
                                                   -r $TEST_RES_REPR_DIR_PATH \
                                                   -m

printf "Training ADOPT on %s \n" $TRAIN_FASTA_FILE_PATH
docker exec -it adopt python ../adopt/training.py -s $TRAIN_STRATEGY \
                                                  -t $TRAIN_JSON_FILE_PATH \
                                                  -e $TEST_JSON_FILE_PATH \
                                                  -r $TRAIN_RES_REPR_DIR_PATH \
                                                  -p $TEST_RES_REPR_DIR_PATH 

docker cp adopt:/ADOPT/models ../
printf "The trained models have been stored in ../models"