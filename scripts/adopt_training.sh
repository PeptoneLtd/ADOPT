# Copyright (c) 2021 Peptone.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#!/bin/bash   

TRAIN_FASTA_FILE_PATH="<train_chezod_fasta_file_path>"
TEST_FASTA_FILE_PATH="<test_chezod_fasta_file_path>"
TRAIN_RES_REPR_DIR_PATH="<train_chezod_residue_level_representation_dir>"
TEST_RES_REPR_DIR_PATH="<test_chezod_residue_level_representation_dir>"
TRAIN_STRATEGY="<training_strategy>"
TRAIN_JSON_FILE_PATH="<train_chezod_json_file_path>"
TEST_JSON_FILE_PATH="<test_chezod_json_file_path>" 

printf "Extracting residue level representation of $TRAIN_FASTA_FILE_PATH and $TEST_FASTA_FILE_PATH \n"

python ../adopt/embedding.py -f $TRAIN_FASTA_FILE_PATH \
                             -r $TRAIN_RES_REPR_DIR_PATH

printf "Training ADOPT on $TRAIN_FASTA_FILE_PATH \n"

python ../adopt/training.py -s $TRAIN_STRATEGY \
                            -t $TRAIN_JSON_FILE_PATH \
                            -e $TEST_JSON_FILE_PATH \
                            -r $TRAIN_RES_REPR_DIR_PATH \
                            -p $TEST_RES_REPR_DIR_PATH 

