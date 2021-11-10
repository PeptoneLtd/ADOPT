import torch
import numpy as np
import constants

# throw away the missing values, if the drop_missing flag is set to True, i.e. where z-scores are  999
def pedestrian_input(indexes, df, path, z_col='z-score', msa=False, drop_missing=True):
    zeds = [] 
    exes = []
    
    if msa==False:
        for k in range(len(indexes)):
            repr_esm = torch.load(f'{path}{indexes[k]}.pt')['representations'][33].clone().cpu().detach()
            z_s = np.array(df[df['brmid']==indexes[k]][z_col].to_numpy()[0])
            if drop_missing==True:
                idxs =  np.where(z_s!=999)[0]
            else:
                idxs = np.arange(len(z_s))
                
            for i in idxs:
                zeds.append(z_s[i])
                exes.append(repr_esm[i].numpy())
    else:
        for k in range(len(indexes)):
            repr_esm = torch.load(f'{path}{indexes[k]}.pt')['representations'].clone().cpu().detach()
            z_s = np.array(df[df['brmid']==indexes[k]][z_col].to_numpy()[0])
            if drop_missing==True:
                idxs =  np.where(z_s!=999)[0]
            else:
                idxs = np.arange(len(z_s))

            for i in idxs:
                zeds.append(z_s[i])
                exes.append(repr_esm[i].numpy())
                
    return np.array(exes), np.array(zeds)

# collect the path to representations according to model type and train vs test set
def representation_path(path_chezod_1325_repr,
                        path_chezod_117_repr):
    repr_path = {}
    for model_type in constants.model_types:
        repr_path[model_type] = {'1325': str(path_chezod_1325_repr)+"/"+model_type+"/",
                                 '117': str(path_chezod_117_repr)+"/"+model_type+"/"}
    return repr_path


def df_to_fasta(df, fasta_out_path):
    ofile = open(fasta_out_path, "w")
    for index, row in df.iterrows():
        ofile.write(">" + row['brmid'] + "\n" + row['sequence'] + "\n")
    ofile.close()
