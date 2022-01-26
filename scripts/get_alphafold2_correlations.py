import json
import pandas as pd
import pickle as pkl
import numpy as np
import torch
from scipy.stats import spearmanr
from subprocess import check_output
import os
import json
import subprocess

def custom_loss(yTrue, yPred, fn=np.mean):
    mask = torch.ne(torch.tensor(yTrue), 999.0)
    yTrue = yTrue[mask]
    yPred = yPred[mask]
    return fn(yTrue,yPred)

def get_model_path(p):
    highest_plddt_model = json.load(open(f'{p}/ranking_debug.json'))['order'][0]
    p = f'{p}/relaxed_{highest_plddt_model}.pdb'
    return p

def get_plddts(p):
    highest_plddt_model = json.load(open(f'{p}/ranking_debug.json'))['order'][0]
    d = pkl.load(open(f'{p}/result_{highest_plddt_model}.pkl','rb'))
    return d['plddt']

def parse_d(d, col):
    d = d[col].apply(pd.Series)
    d.columns = [f'{col}-{c}' for c in d.columns]
    return d

def get_relative_areas(d):
    d = d['results'][0]['structure'][0]['chains'][0]['residues']
    d = pd.DataFrame(d)
    d = pd.concat([d]+[parse_d(d, col) for col in ['area', 'relative-area']], axis=1)
    return d['relative-area-total'].to_numpy()

def get_correlation(model_folder, experimental_path):
    f = model_folder
    df = pd.read_json(experimental_path)

    df['plddt'] = df.brmid.apply(lambda idx: np.array(get_plddts(f'{f}/{idx}/')))
    df['model_path'] = df.brmid.apply(lambda idx: np.array(get_model_path(f'{f}/{idx}/')))
    df['sasa_json'] = df.model_path.apply(lambda s: check_output(f'freesasa {s} --depth=residue --format=json', shell=True) )
    df['sasa_dict'] = df.sasa_json.apply(lambda s: json.loads(s.decode()))
    df['rel_sasa'] = df['sasa_dict'].apply(get_relative_areas)
    df_ = df

    cols = []
    for col in ['plddt','rel_sasa']:
        for n in np.arange(5, 30, 5):
            cols.append(f'{col}_{n}_roll')
            df[f'{col}_{n}_roll'] = df[col].apply(lambda x: pd.Series(x).rolling(n, min_periods=0, center=True).mean().to_numpy())

    cols = list(set(['zscore', 'plddt','rel_sasa']+cols))
    s = [df[col].apply(pd.Series).set_index(df.brmid).stack() for col in cols]
    df = pd.concat(s, axis=1)
    df.columns = cols

    results = {col: custom_loss(df.zscore.to_numpy(), df[col].to_numpy(), fn=spearmanr).correlation for col in set(cols).union(set(['plddt','rel_sasa']))-set(['zscore'])}
    r = pd.DataFrame([results])
    print(r[sorted(r.columns)].T.rename(columns={0:"Spearman correlation to Z-score"}))


f = '/data/structure_predictions'
subprocess.call('tar -xvf /data.tar.gz -C /', shell=True)
get_correlation(f,'/data/117_dataset_raw.json')