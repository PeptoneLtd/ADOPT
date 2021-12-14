#!/bin/bash 

# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

TRAIN_FASTA_FILE_PATH="../datasets/chezod_1325_all.fasta"
TEST_FASTA_FILE_PATH="../datasets/chezod_117_all.fasta"
TRAIN_RES_REPR_DIR_PATH="../representations/chezod_1325"
TEST_RES_REPR_DIR_PATH="../representations/chezod_117"
BENCHMARK_DATA_PATH="<benchmark_prediction_path>" # path of the predictions from the method you want to benchmark
TRAIN_STRATEGY="<training_strategy>" # choose a training strategy
TRAIN_JSON_FILE_PATH="../datasets/1325_dataset_raw.json"
TEST_JSON_FILE_PATH="../datasets/117_dataset_raw.json" 

printf "Extracting residue level representation of %s \n" $TRAIN_FASTA_FILE_PATH

python adopt/embedding.py $TRAIN_FASTA_FILE_PATH \
                          $TRAIN_RES_REPR_DIR_PATH

printf "Extracting residue level representation of %s \n" $TEST_FASTA_FILE_PATH

python adopt/embedding.py $TEST_FASTA_FILE_PATH \
                          $TEST_RES_REPR_DIR_PATH

printf "Benchmarking predictions in %s \n" $BENCHMARK_DATA_PATH
printf "The results are stored in ../media \n"

python adopt/benchmarks.py $BENCHMARK_DATA_PATH \
                           $TRAIN_JSON_FILE_PATH \
                           $TEST_JSON_FILE_PATH \
                           $TRAIN_RES_REPR_DIR_PATH \
                           $TEST_RES_REPR_DIR_PATH \
                           --train_strategy $TRAIN_STRATEGY