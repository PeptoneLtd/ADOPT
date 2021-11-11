# Attention DisOrder  PredicTor

This repository containes the code and the trained models for **intrinsic protein disorder prediction through deep bidirectional transformers** from Peptone Ltd.

Adopt has been introduced in our paper [link](ADOPT: intrinsic protein disorder prediction throughdeep bidirectional transformers).

Our disorder predictor is made up of two main blocks, namely: a **self-supervised encoder* and a **supervised disorder predictor**. We use [https://github.com/facebookresearch/esm](Facebook’s Evolutionary Scale Modeling) library to extract dense residue evel representations, which feed the  supervised machine learning based predictor. 

The ESM library exploits a set of deep Transformer encoder models, which processes character sequences of amino acids as inputs.

ADOPT makes use of two datasets: the [CheZoD  “1325” and the CheZoD “117”](https://github.com/protein-nmr/CheZOD) databases containing 1325 and 117 sequences, respectively, together with their  residue level **Z-scores**.

## Intrinsic disorder trained models

| Model | Pre-trained model | Datasets | Split level | CV |
|-------|-------------------|----------|-------------|----|
| `lasso_esm-1b_cleared_residue` | ESM-1b | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_esm-1v_cleared_residue` | ESM-1v | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_combined_cleared_residue` | Combined | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_esm-1b_residue_cv` | ESM-1b | **Chezod 1325** | residue | :white_check_mark: |
| `lasso_esm-1v_residue_cv` | ESM-1v | **Chezod 1325** | residue | :white_check_mark: |
| `lasso_esm-1b_cleared_residue_cv` | ESM-1b | **Chezod 1325 cleared** | residue | :white_check_mark: |
| `lasso_esm-1v_cleared_residue_cv` | ESM-1v | **Chezod 1325 cleared** | residue | :white_check_mark: |
| `lasso_esm-1b_cleared_sequence_cv` | ESM-1b | **Chezod 1325 cleared** | residue | :white_check_mark: |
| `lasso_esm-1v_cleared_sequence_cv` | ESM-1v | **Chezod 1325 cleared** | sequence | :white_check_mark: |

## Usage

### Quick start

As a prerequisite, you must have fair-esm 0.4 or later installed to use this repository.

Install the **adopt** package:

Clone the ADOPT repository, go to the ADOPT directory and run

```bash
$ python setup.py install
```

### Compute residue level representations

In order to predict the **Z score** related to each residue in a protein sequence, we have to compute the residue level representations, extracted from the pretrained model. 

In the ADOPT directory run:

```bash
$ python embedding.py -f <fasta_file_path> -r <residue_level_representation_dir>
```

Where:
* `-f` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
* `-r` defines the path where you want to save the residue level representations

A subdirectory containing the residue level representation extracted from each pre-trained model available will be created under both the `residue_level_representation_dir`.


### Predict intrinsic disorder with ADOPT

Once we have extracted the residue level representations we can predict the intrinsic disorder (Z score).

In the ADOPT directory run:

```bash
$ inference.py s- <training_strategy> -m <model_type=> -f <inference_fasta_file=> -r <inference_repr_dir> -p <predicted_z_scores_file>
```

Where:
* `-s` defines the **training strategies** defined belowe
* `-f` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
* `-r` defines the path where you've already saved the residue level representations
* `-p` defines the path where you want the Z scores to be saved

The output is a `.csv`file contains the Z scores related to each residue of each protein in the FASTA file where you put the proteins you are intereseted in.

| Training strategy | CV |
|-------------------|----|
| `train_on_cleared_1325_test_on_117_residue_split` | :x: |
| `train_on_1325_cv_residue_split`| :white_check_mark: |
| `train_on_cleared_1325_cv_residue_split`| :white_check_mark: |
| `train_on_cleared_1325_cv_sequence_split`| :white_check_mark: |


### Train ADOPT disorder predictor

Once we have extracted the residue level representations of the protein for which we want to predict the intrinsic disorder (Z score), we can train the predictor.

**NOTE**: This step is not mandatory because we've already trained such models. You can find them in the *models* directory.

In the ADOPT directory run:

```bash
$ python training.py s- <training_strategy> -t <train_json_file_path=> -e <test_json_file_path=> -r <train_residue_level_representation_dir> -p <test_residue_level_representation_dir>
```

Where:
* `-s` defines the **training strategies** defined above
* `-t` defines the JSON containing the proteins we want to use as *training set*
* `-e` defines the JSON containing the proteins we want to use as *test set*
* `-r` defines the path where we saved the residue level representations of the proteins in the *training set*
* `-p` defines the path where we saved the residue level representations of the proteins in the *test set*

