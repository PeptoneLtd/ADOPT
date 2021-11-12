#!/bin/bash   

TRAIN_FASTA_FILE_PATH="<train_chezod_fasta_file_path>"
TEST_FASTA_FILE_PATH="<test_chezod_fasta_file_path>"
TRAIN_RES_REPR_DIR_PATH="<train_chezod_residue_level_representation_dir>"
TEST_RES_REPR_DIR_PATH="<test_chezod_residue_level_representation_dir>"
TRAIN_STRATEGY="<training_strategy>"
TRAIN_JSON_FILE_PATH="<train_chezod_json_file_path>"
TEST_JSON_FILE_PATH="<test_chezod_json_file_path>" 


python embedding.py -f $TRAIN_FASTA_FILE_PATH 
                    -r $TRAIN_RES_REPR_DIR_PATH

printf "Extracting the residue level representation of CheZod training and test set"

python training.py -s $TRAIN_STRATEGY
                   -t $TRAIN_JSON_FILE_PATH 
                   -e $TEST_JSON_FILE_PATH 
                   -r $TRAIN_RES_REPR_DIR_PATH 
                   -p $TEST_RES_REPR_DIR_PATH

printf "Training ADOPT on CheZod"
