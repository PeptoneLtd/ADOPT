#!/bin/bash 

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
 

TRAIN_FASTA_FILE_PATH="../datasets/chezod_1325_all.fasta"
TEST_FASTA_FILE_PATH="../datasets/chezod_117_all.fasta"
TRAIN_RES_REPR_DIR_PATH="../representations/chezod_1325"
TEST_RES_REPR_DIR_PATH="../representations/chezod_117"
TRAIN_STRATEGY="<training_strategy>" # choose a training strategy
TRAIN_JSON_FILE_PATH="../datasets/1325_dataset_raw.json"
TEST_JSON_FILE_PATH="../datasets/117_dataset_raw.json" 

printf "Extracting residue level representation of %s \n" $TRAIN_FASTA_FILE_PATH

python ../adopt/embedding.py -f $TRAIN_FASTA_FILE_PATH \
                             -r $TRAIN_RES_REPR_DIR_PATH

printf "Extracting residue level representation of %s \n" $TEST_FASTA_FILE_PATH

python ../adopt/embedding.py -f $TEST_FASTA_FILE_PATH \
                             -r $TEST_RES_REPR_DIR_PATH

printf "Training ADOPT on %s \n" $TRAIN_FASTA_FILE_PATH

python ../adopt/training.py -s $TRAIN_STRATEGY \
                            -t $TRAIN_JSON_FILE_PATH \
                            -e $TEST_JSON_FILE_PATH \
                            -r $TRAIN_RES_REPR_DIR_PATH \
                            -p $TEST_RES_REPR_DIR_PATH 


