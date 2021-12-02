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
    - [MSA setting (optional)](#msa-setting-optional)
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

### MSA setting (optional)

In order to enable the ```esm-msa``` based variant of ADOPT, MSAs for each sequence are also required.
We provide a stand alone, ```docker``` based tool that can be used to obatin MSAs from ```fasta``` files.
If the user already has her/his MSAs ready, the steps to follow will be given below.

#### Setup the tool to generate MSAs

Make the following three scripts in the [scripts](scripts) folder locally available:
- ```adopt_msa_setup.sh``` [adopt_msa_setup](scripts/adopt_msa_setup.sh);
- ```uniclust_download.py``` [adopt_msa_setup](scripts/uniclust_download.py);
- ```msa_generator.sh``` [adopt_msa_setup](scripts/msa_generator.sh);

This means if ADOPT is running in a container, these files should be copied into a local host folder and
not used directly in the container.

<code> <b> STEP 1 </b> </code>

Assuming that the above three files are placed in a local folder ```/tmp```, the first step is to run,
from ```/tmp```, the following command <code><i> (Please read the description before run!) </i></code>

```bash
adopt_msa_setup.sh <local_msa_dir>
```

where ```local_msa_dir``` serves as the main directory for the MSA related procedures and can be empty
initially when running the above script. This takes care of two tasks:

 1. Downloading [uniclust](http://gwdu111.gwdg.de/~compbiol/uniclust/) dataset (in this case "2020.06") into
  the ```/local_msa_dir/databases``` subdirectory. <code><i> (NOTE!) </i></code> Under the hood,
<code>uniclust_download.py</code> runs and checks, whether uniclust is already in this subdirectory. If not, downloading
can take several hours, given the size of this dataset is approx 180GB! Download step is skipped only if the ```/local_msa_dir/databases```
folder is non empty and the tar file (<cite>UniRef30_2020_06_hhsuite.tar.gz</cite>) is found in the ```/local_msa_dir``` folder.

 2. Once the relevant uniclust is there, a docker image named ```msa-gen-adopt``` is run with the volume ```/local_msa_dir```
mounted on it.

Note that this setup procedure creates four subfolders:

<pre>
+-- local_msa_dir
|   +-- databases
|   +-- msas
|   +-- msa_fastas
|   +-- esm_msa_reprs
</pre>

```databases``` will hold the [uniclust](http://gwdu111.gwdg.de/~compbiol/uniclust/);
```msas``` is where MSAs (```.a3m``` files) will be saved later, see STEP 2 below;
```msa_fastas``` is where ```.fasta``` files already used for MSA queries will be saved;
```esm_msa_reprs``` is allocated for potential ```esm-msa``` representations;

<code> <b> STEP 2 </b> </code>

Given the docker container ```msa-gen-adopt``` is up and running (the result of STEP 1), the following command can be used,
from the ```/tmp``` folder, to generate MSAs using a fasta file:

```bash
msa_generator.sh <fasta_file_path>
```
The MSAs will be placed in the ```/local_msa_dir/msas``` folder. Furthermore, the fasta file used for query will be copied in the
```/local_msa_dir/msa_fastas``` folder.

### Scripts

The [scripts](scripts) directory contains:

- [inference](scripts/adopt_inference.sh) script to predict, in bulk, the disorder of each residue in each protein sequence reported in a FASTA file, with ADOPT where you need to specify:
  - `NEW_PROT_FASTA_FILE_PATH` defining your FASTA file path
  - `NEW_PROT_RES_REPR_DIR_PATH` defining where the residue level representations will be extracted
- [training](scripts/adopt_chezod_training.sh) script to train the ADOPT where you need to specify:
  - `TRAIN_STRATEGY` defining the training strategy you want to use

### Notebooks

The [notebooks](notebooks) directory contains:

- [disorder prediction](notebooks/adopt_disorder_prediction.ipynb) notebook
- [multi-head attention weights visualisation](notebooks/adopt_attention_viz.ipynb) notebook

### Compute residue level representations

In order to predict the **Z score** related to each residue in a protein sequence, we have to compute the residue level representations, extracted from the pretrained model.

In the ADOPT directory run:

```bash
python embedding.py -f <fasta_file_path> \
                    -r <residue_level_representation_dir>
```

Where:

- `-f` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
- `-r` defines the path where you want to save the residue level representations
- `-h` shows help message and exit

A subdirectory containing the residue level representation extracted from each pre-trained model available will be created under both the `residue_level_representation_dir`.

Important to note that in order to obtain the representations from the ```esm-msa``` model as well, the relevant MSAs have to
be placed in the root directory `/msas` in the system, where ADOPT is running. The MSAs can be created as described in
the MSA setting above.
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

- `-s` defines the **training strategies** defined below
- `-m` defines the pre-trained model we want to use. We suggest you use the `esm-1b` model.
- `-f` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
- `-r` defines the path where you've already saved the residue level representations
- `-p` defines the path where you want the Z scores to be saved
- `-h` shows help message and exit

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

- `-s` defines the **training strategies** defined above
- `-t` defines the JSON containing the proteins we want to use as *training set*
- `-e` defines the JSON containing the proteins we want to use as *test set*
- `-r` defines the path where we saved the residue level representations of the proteins in the *training set*
- `-p` defines the path where we saved the residue level representations of the proteins in the *test set*
- `-h` shows help message and exit

## Citations <a name="citations"></a>

If you use this work in your research, please cite the the relevant paper:

```bibtex
@article{redl2021adopt}
```

## Licence

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.
