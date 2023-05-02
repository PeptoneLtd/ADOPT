# Attention based DisOrder PredicTor

This repository containes the code and the trained models for **intrinsic protein disorder prediction through deep bidirectional transformers** from Peptone Ltd.

[![DOI](https://zenodo.org/badge/426679803.svg)](https://zenodo.org/badge/latestdoi/426679803)
[![GitHub Super-Linter](https://github.com/peptoneltd/ADOPT/workflows/Lint%20Code%20Base/badge.svg)](https://github.com/marketplace/actions/super-linter)

ADOPT has been introduced in our paper [ADOPT: intrinsic protein disorder prediction through deep bidirectional transformers](https://www.biorxiv.org/content/10.1101/2022.05.25.493416v3) and it's also available as webserver at [adopt.peptone.io](https://adopt.peptone.io).

<img src="https://github.com/PeptoneLtd/ADOPT/blob/main/media/adopt_attention.gif" width="600"/>

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
    - [Run benchmarks](#run-benchmarks)
  - [Citations](#citations)
  - [Licence](#licence)

## Intrinsic disorder trained models

| Model | Pre-trained model | Datasets | Split level | CV |
|-------|-------------------|----------|-------------|----|
| `lasso_esm-1b_cleared_residue` | ESM-1b | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_esm-1v_cleared_residue` | ESM-1v | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_esm-msa_cleared_residue` | ESM-MSA | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_combined_cleared_residue` | Combined | **Chezod 1325 cleared** and **Chezod 117** | residue | :x: |
| `lasso_esm-1b_residue_cv` | ESM-1b | **Chezod 1325** | residue | :white_check_mark: |
| `lasso_esm-1v_residue_cv` | ESM-1v | **Chezod 1325** | residue | :white_check_mark: |
| `lasso_esm-msa_residue_cv` | ESM-MSA | **Chezod 1325** | residue | :white_check_mark: |
| `lasso_esm-1b_cleared_residue_cv` | ESM-1b | **Chezod 1325 cleared** | residue | :white_check_mark: |
| `lasso_esm-1v_cleared_residue_cv` | ESM-1v | **Chezod 1325 cleared** | residue | :white_check_mark: |
| `lasso_esm-msa_cleared_residue_cv` | ESM-MSA | **Chezod 1325 cleared** | residue | :white_check_mark: |
| `lasso_esm-1b_cleared_sequence_cv` | ESM-1b | **Chezod 1325 cleared** | sequence | :white_check_mark: |
| `lasso_esm-1v_cleared_sequence_cv` | ESM-1v | **Chezod 1325 cleared** | sequence | :white_check_mark: |
| `lasso_esm-msa_cleared_sequence_cv` | ESM-MSA | **Chezod 1325 cleared** | sequence | :white_check_mark: |

## Usage

### Quick start

Prerequisites (we suggest creating a dedicated python venv or conda env)

```
pip install \
  pandas \
  fair-esm \
  biopython \
  bertviz \
  skl2onnx \
  onnxruntime \
  spacy \
  plotly \
  wandb 
```

Install the **adopt** package:

Run

```bash
git clone https://github.com/PeptoneInc/ADOPT.git
cd ADOPT
git submodule update --init --recursive
python setup.py install
```

Then, you can predict the intrinsic disorder of each reesidue in a protein sequence, as follows:

```python
from adopt import MultiHead, ZScorePred

# Prepare protein sequence and name i.e brmid
SEQUENCE = "SLQDGVRQSRASDKQTLLPNDQLYQPLKDREDDQYSHLQGNQLRRN"
PROTID = "Protein 18890"

# Choose model type and training strategy
MODEL_TYPE = "esm-1b"
STRATEGY = "train_on_cleared_1325_test_on_117_residue_split"

# Extract residue level representations
multi_head = MultiHead(MODEL_TYPE)
representation, tokens = multi_head.get_representation(SEQUENCE, PROTID)

# Predict the Z score related to each residue in the sequence specified above
z_score_pred = ZScorePred(STRATEGY, MODEL_TYPE)
predicted_z_scores = z_score_pred.get_z_score(representation)
````

### MSA setting (optional)

In order to enable the ```esm-msa``` based variant of ADOPT, MSAs for each sequence are also required.
We provide a stand alone, ```docker``` based tool you must use to exploit all the functionalities of ADOPT for `msa` related tasks.

#### First time setup

As a prerequisite, you must have [Docker](https://www.docker.com/) installed.

Clone the ADOPT repository, go to the ADOPT directory and run the [MSA scripts](#scripts) you are interested in.

#### Notes

The `$LOCAL_MSA_DIR` in the **MSA scripts** serves as the main directory for the MSA related procedures and can be empty
initially when running the above scripts. Under the hood, each **MSA script** will:

 1. Download [uniclust](http://gwdu111.gwdg.de/~compbiol/uniclust/) dataset (in this case "2020.06") into
  the ```$LOCAL_MSA_DIR/databases``` subdirectory. \
  **!NOTE**: under the hood,
  [ADOPT](scripts/uniclust_download.py) checks, whether uniclust is already in this subdirectory. If not, downloading
  can take several hours, given the size of this dataset is approx 180GB! Download step is skipped only if the ```$LOCAL_MSA_DIR/databases```
  folder is non empty and the tar file (<cite>UniRef30_2020_06_hhsuite.tar.gz</cite>) is found in the ```$LOCAL_MSA_DIR``` folder.

 2. Once the relevant uniclust is there, a docker image named ```msa-gen-adopt``` is run with the volume ```$LOCAL_MSA_DIR```
mounted on it.

Note that this setup procedure creates four subfolders:

<pre>
+-- $LOCAL_MSA_DIR
|   +-- databases
|   +-- msas
|   +-- msa_fastas
|   +-- esm_msa_reprs
</pre>

```databases``` will hold the [uniclust](http://gwdu111.gwdg.de/~compbiol/uniclust/);
```msas``` is where MSAs (```.a3m``` files) will be saved later, see STEP 2 below;
```msa_fastas``` is where ```.fasta``` files already used for MSA queries will be saved;
```esm_msa_reprs``` is allocated for potential ```esm-msa``` representations;

The MSAs will be placed in the ```$LOCAL_MSA_DIR/msas``` folder.

### More notes

You can set the ```ESM_MODELS_DIR``` and ```ADOPT_MODELS_DIR``` respectively to paths where the [ESM](https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt) and [ADOPT](https://adopt-models.s3.eu-west-2.amazonaws.com/models.zip) pretrained models are stored.
All models will be downloaded from public repositories if not found locally.

### Scripts

The [scripts](scripts) directory contains:

- [inference](scripts/adopt_inference.sh) script to predict, in bulk, the disorder of each residue in each protein sequence reported in a FASTA file, with ADOPT where you need to specify:
  - `NEW_PROT_FASTA_FILE_PATH` defining your FASTA file path
  - `NEW_PROT_RES_REPR_DIR_PATH` defining where the residue level representations will be extracted
- [training](scripts/adopt_chezod_training.sh) script to train the ADOPT where you need to specify:
  - `TRAIN_STRATEGY` defining the training strategy you want to use
- [MSA inference](scripts/adopt_msa_inference.sh) script, which allows to perform [inference](scripts/adopt_inference.sh) also with the `esm-msa` model. The predicted Z scores will be written on the host (**optional**)
- [MSA training](scripts/adopt_chezod_msa_training.sh) script, which allows to perform [training](scripts/adopt_chezod_training.sh) also with the `esm-msa` model. The trained models will be written in the `ADOPT/models` directory (**optional**)

### Notebooks

The [notebooks](notebooks) directory contains:

- [disorder prediction](notebooks/adopt_disorder_prediction.ipynb) notebook
- [multi-head attention weights visualisation](notebooks/adopt_attention_viz.ipynb) notebook

### Compute residue level representations

In order to predict the **Z score** related to each residue in a protein sequence, we have to compute the residue level representations, extracted from the pretrained model.

In the ADOPT directory run:

```bash
python adopt/embedding.py <fasta_file_path> \
                          <residue_level_representation_dir>
```

Where:

- `<fasta_file_path>` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
- `<residue_level_representation_dir>` defines the path where you want to save the residue level representations
- `--msa` runs the [MSA procedure](#msa-setting-optional) to get `esm-msa` representations. We suggest you take a look to the [MSA inference](scripts/adopt_msa_inference.sh) script as a quick example (**optional**)
- `-h` shows help message and exit

A subdirectory containing the residue level representation extracted from each pre-trained model available will be created under both the `residue_level_representation_dir`.

Important to note that in order to obtain the representations from the ```esm-msa``` model as well, the relevant MSAs have to
be placed in the root directory `/msas` in the system, where ADOPT is running. The MSAs can be created as described in
the MSA setting above.

### Predict intrinsic disorder with ADOPT

Once we have extracted the residue level representations we can predict the intrinsic disorder (Z score).

In the ADOPT directory run:

```bash
python adopt/inference.py <inference_fasta_file> \
                          <inference_repr_dir> \
                          <predicted_z_scores_file> \
                          --train_strategy <training_strategy> \
                          --model_type <model_type> 
```

Where:

- `<inference_fasta_file>` defines the FASTA file containing the proteins for which you want to compute the intrinsic disorder
- `<inference_repr_dir>` defines the path where you've already saved the residue level representations
- `<predicted_z_scores_file>` defines the path where you want the Z scores to be saved
- `--train_strategy` defines the **training strategies** defined below
- `--model_type` defines the pre-trained model we want to use. We suggest you use the `esm-1b` model
- `-h` shows help message and exit

The output is a `.json` file contains the Z scores related to each residue of each protein in the FASTA file where you put the proteins you are intereseted in.

| Training strategy | Pre-trained models |
|-------------------|-------------------|
| `train_on_cleared_1325_test_on_117_residue_split` | `esm-1b`, `esm-1v`, `esm-msa` and `combined` |
| `train_on_1325_cv_residue_split`| `esm-1b`, `esm-1v` and `esm-msa` |
| `train_on_cleared_1325_cv_residue_split`| `esm-1b`, `esm-1v` and `esm-msa` |
| `train_on_cleared_1325_cv_sequence_split`| `esm-1b`, `esm-1v` and `esm-msa` |
| `train_on_total`| `esm-1b`, `esm-1v`|

### Train ADOPT disorder predictor

Once we have extracted the residue level representations of the protein for which we want to predict the intrinsic disorder (Z score), we can train the predictor.

**NOTE**: This step is not mandatory because we've already trained such models. You can find them in the [models](https://adopt-models.s3.eu-west-2.amazonaws.com/models.zip) bucket.

In the ADOPT directory run:

```bash
python adopt/training.py <train_json_file_path> \
                         <test_json_file_path> \
                         <train_residue_level_representation_dir> \
                         <test_residue_level_representation_dir> \
                         --train_strategy <training_strategy> 
```

Where:

- `<train_json_file_path>` defines the JSON containing the proteins we want to use as *training set*
- `<test_json_file_path>` defines the JSON containing the proteins we want to use as *test set*
- `<train_residue_level_representation_dir>` defines the path where we saved the residue level representations of the proteins in the *training set*
- `<test_residue_level_representation_dir>` defines the path where we saved the residue level representations of the proteins in the *test set*
- `--train_strategy` defines the **training strategies** defined above
- `--msa` runs the [MSA procedure](msa-setting-optional) to get trained models fed with the `esm-msa` representations. We suggest you take a look to the [MSA training](scripts/adopt_chezod_msa_training.sh) script as a quick example (**optional**)
- `-h` shows help message and exit

### Run benchmarks

Once we have extracted the residue level representations we can benchmark ADOPT against other methods.

In the ADOPT directory run:

```bash
python adopt/benchmarks.py <benchmark_data_path> \
                           <train_json_file_path> \
                           <test_json_file_path> \
                           <train_residue_level_representation_dir> \
                           <test_residue_level_representation_dir> \
                           --train_strategy <training_strategy> 
```

Where:

- `<benchmark_data_path>` defines the directory containing the predictions of the method we want to benchmark againbst ADOPT
- `-h` shows help message and exit

#### AlphaFold2 benchmarks (optional)

We benchmarked ADOPT against [AlphaFold2](https://github.com/deepmind/alphafold) computing the spearman correlations between actual Z-scores and predicted pLDDT<sub>5</sub> scores along with actual Z-scores and predicted SASA<sub>5</sub> scores,
obtained by AlphaFold2, collected for the task linked to the model evaluated on the [CheZoD “117”](datasets/chezod_117_all.fasta) validation set and described in the ADOPT [paper](https://link_to_paper).

As a prerequisite, you must have [Docker](https://www.docker.com/) installed.

Run:

```bash
docker run ghcr.io/peptoneinc/adopt_alphafold2_comparison:1.0.2
```

[Here](scripts/get_alphafold2_correlations.py) is the script used to extract the correlations and [here](https://alphafold2-chezod-predictions.s3.eu-central-1.amazonaws.com/data.tar.gz) are the predictions obtained from Alphafold2.

## Citations <a name="citations"></a>

If you use this work in your research, please cite the the relevant paper:

```bibtex
@article{10.1093/nargab/lqad041,
    author = {Redl, Istvan and Fisicaro, Carlo and Dutton, Oliver and Hoffmann, Falk and Henderson, Louie and Owens, Benjamin M J and Heberling, Matthew and Paci, Emanuele and Tamiola, Kamil},
    title = "{ADOPT: intrinsic protein disorder prediction through deep bidirectional transformers}",
    journal = {NAR Genomics and Bioinformatics},
    volume = {5},
    number = {2},
    year = {2023},
    month = {05},
    abstract = "{Intrinsically disordered proteins (IDPs) are important for a broad range of biological functions and are involved in many diseases. An understanding of intrinsic disorder is key to develop compounds that target IDPs. Experimental characterization of IDPs is hindered by the very fact that they are highly dynamic. Computational methods that predict disorder from the amino acid sequence have been proposed. Here, we present ADOPT (Attention DisOrder PredicTor), a new predictor of protein disorder. ADOPT is composed of a self-supervised encoder and a supervised disorder predictor. The former is based on a deep bidirectional transformer, which extracts dense residue-level representations from Facebook’s Evolutionary Scale Modeling library. The latter uses a database of nuclear magnetic resonance chemical shifts, constructed to ensure balanced amounts of disordered and ordered residues, as a training and a test dataset for protein disorder. ADOPT predicts whether a protein or a specific region is disordered with better performance than the best existing predictors and faster than most other proposed methods (a few seconds per sequence). We identify the features that are relevant for the prediction performance and show that good performance can already be gained with \\&lt;100 features. ADOPT is available as a stand-alone package at https://github.com/PeptoneLtd/ADOPT and as a web server at https://adopt.peptone.io/.}",
    issn = {2631-9268},
    doi = {10.1093/nargab/lqad041},
    url = {https://doi.org/10.1093/nargab/lqad041},
    note = {lqad041},
    eprint = {https://academic.oup.com/nargab/article-pdf/5/2/lqad041/50150244/lqad041.pdf},
}
```

## Licence

This source code is licensed under the MIT license found in the `LICENSE` file in the root directory of this source tree.
