# Attention based DisOrder PredicTor

This repository containes the code and the trained models for **intrinsic protein disorder prediction through deep bidirectional transformers** from Peptone Ltd.

[![GitHub Super-Linter](https://github.com/peptoneinc/ADOPT/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

ADOPT has been introduced in our paper [ADOPT: intrinsic protein disorder prediction throughdeep bidirectional transformers](link).

<img src="https://github.com/PeptoneInc/ADOPT/blob/main/media/adopt_attention.gif" width="600"/>

Our disorder predictor is made up of two main blocks, namely: a **self-supervised encoder** and a **supervised disorder predictor**. We use [Facebook’s Evolutionary Scale Modeling (ESM)](https://github.com/facebookresearch/esm) library to extract dense residue evel representations, which feed the  supervised machine learning based predictor.

The ESM library exploits a set of deep Transformer encoder models, which processes character sequences of amino acids as inputs.

ADOPT makes use of two datasets: the [CheZoD  “1325” and the CheZoD “117”](https://github.com/protein-nmr/CheZOD) databases containing 1325 and 117 sequences, respectively, together with their  residue level **Z-scores**.

## Table of Contents

- [Attention based DisOrder PredicTor](#attention-based-disorder-predictor)
  - [Table of Contents](#table-of-contents)
  - [Intrinsic disorder trained models](#intrinsic-disorder-trained-models)
  - [Usage](#usage)
    - [Quick start](#quick-start)
    - [Scripts](#scripts)
    - [Notebooks](#notebooks)
    - [Compute residue level representations](#compute-residue-level-representations)
    - [Predict intrinsic disorder with ADOPT](#predict-intrinsic-disorder-with-adopt)
    - [Train ADOPT disorder predictor](#train-adopt-disorder-predictor)
  - [Citations](#citations)
  - [Licence](#licence)

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

As a prerequisite, you must have [PyTorch 1.10](https://pytorch.org/) or later installed to use this repository.

Install the **adopt** package:

Clone the ADOPT repository, go to the ADOPT directory and run

```bash
python setup.py install
```

Then, you can predict the intrinsic disorder of each reesidue in a protein sequence, as follows:

```python
from adopt import MultiHead, ZScorePred

# Prepare protein sequence and name i.e brmid 
SEQUENCE = "SLQDGVRQSRASDKQTLLPNDQLYQPLKDREDDQYSHLQGNQLRRN"
BRMID = "Protein 18890"

# Choose model type and training strategy
MODEL_TYPE = "esm-1b"
STRATEGY = "train_on_cleared_1325_test_on_117_residue_split"

# Extract residue level representations
multi_head = MultiHead(MODEL_TYPE, SEQUENCE, BRMID)
representation, tokens = multi_head.get_representation()

# Predict the Z score related to each residue in the sequence specified above
z_score_pred = ZScorePred(STRATEGY, MODEL_TYPE)
predicted_z_scores = z_score_pred.get_z_score(representation)
````

### Scripts

The [scripts](scripts) directory contains:

* [inference](scripts/adopt_inference.sh) script to predict, in bulk, the disorder of each residue in each protein sequence reported in a FASTA file, with ADOPT where you need to specify:
  - `NEW_PROT_FASTA_FILE_PATH` defining your FASTA file path
  - `NEW_PROT_RES_REPR_DIR_PATH` defining where the residue level representations will be extracted  
* [training](scripts/adopt_chezod_training.sh) script to train the ADOPT where you need to specify:
  - `TRAIN_STRATEGY` defining the training strategy you want to use

### Notebooks

The [notebooks](notebooks) directory contains:

* [disorder prediction](notebooks/adopt_disorder_prediction.ipynb) notebook
* [multi-head attention weights visualisation](notebooks/adopt_attention_viz.ipynb) notebook

### Compute residue level representations

In order to predict the **Z score** related to each residue in a protein sequence, we have to compute the residue level representations, extracted from the pretrained model.

In the ADOPT directory run:

```bash
python embedding.py -f <fasta_file_path> \
                    -r <residue_level_representation_dir> 
```

Where:

* `-f` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
* `-r` defines the path where you want to save the residue level representations
* `-h` shows help message and exit

A subdirectory containing the residue level representation extracted from each pre-trained model available will be created under both the `residue_level_representation_dir`.

### Predict intrinsic disorder with ADOPT

Once we have extracted the residue level representations we can predict the intrinsic disorder (Z score).

In the ADOPT directory run:

```bash
python inference.py -s <training_strategy> \
                    -m <model_type> \
                    -f <inference_fasta_file> \
                    -r <inference_repr_dir> \
                    -p <predicted_z_scores_file>
```

Where:

* `-s` defines the **training strategies** defined below
* `-m` defines the pre-trained model we want to use. We suggest you use the `esm-1b` model.
* `-f` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
* `-r` defines the path where you've already saved the residue level representations
* `-p` defines the path where you want the Z scores to be saved
* `-h` shows help message and exit

The output is a `.json` file contains the Z scores related to each residue of each protein in the FASTA file where you put the proteins you are intereseted in.

| Training strategy | Pre-trained models |
|-------------------|-------------------|
| `train_on_cleared_1325_test_on_117_residue_split` | `esm-1b`, `esm-1v` and `combined` |
| `train_on_1325_cv_residue_split`| `esm-1b` and `esm-1v` |
| `train_on_cleared_1325_cv_residue_split`| `esm-1b` and `esm-1v` |
| `train_on_cleared_1325_cv_sequence_split`| `esm-1b` and `esm-1v` |

### Train ADOPT disorder predictor

Once we have extracted the residue level representations of the protein for which we want to predict the intrinsic disorder (Z score), we can train the predictor.

**NOTE**: This step is not mandatory because we've already trained such models. You can find them in the *models* directory.

In the ADOPT directory run:

```bash
python training.py -s <training_strategy> \
                   -t <train_json_file_path> \
                   -e <test_json_file_path> \
                   -r <train_residue_level_representation_dir> \
                   -p <test_residue_level_representation_dir>
```

Where:

* `-s` defines the **training strategies** defined above
* `-t` defines the JSON containing the proteins we want to use as *training set*
* `-e` defines the JSON containing the proteins we want to use as *test set*
* `-r` defines the path where we saved the residue level representations of the proteins in the *training set*
* `-p` defines the path where we saved the residue level representations of the proteins in the *test set*
* `-h` shows help message and exit

## Citations <a name="citations"></a>

If you use this work in your research, please cite the the relevant paper:

```bibtex
@article{redl2021adopt}
```

## Licence

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.
