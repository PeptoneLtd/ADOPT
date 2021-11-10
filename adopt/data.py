import pandas as pd
import constants
import utils


class CheZod:
    def __init__(self, path_chezod_1325_raw, path_chezod_117_raw):
        self.path_chezod_1325_raw = str(path_chezod_1325_raw)
        self.path_chezod_117_raw = str(path_chezod_117_raw)

    def get_chezod_raw(self):
        df_ch = pd.read_json(self.path_chezod_1325_raw)
        df_117 = pd.read_json(self.path_chezod_117_raw)

        # since there are some proteins in the 1325 set, we will remove these and create a reduced dataframe for later use 
        # check the overlap, if any exists, in the 117 and 1325 sets
        overlaps = list(set(list(df_ch['brmid'])) & set(list(df_117['brmid'])))

        # Drop the overlaps from the 1325 
        df_cleared = df_ch[~df_ch['brmid'].isin(overlaps)]
        return df_cleared, df_ch, df_117

    def get_train_test_sets(self,
                            path_chezod_1325_repr,
                            path_chezod_117_repr):
        # collect the path to representations according to model type and train vs test set
        repr_path = utils.representation_path(path_chezod_1325_repr,
                                              path_chezod_117_repr)

        df_cleared, _, df_117 = self.get_chezod_raw()

        # read the data 
        ex_train, zed_train = {}, {}
        ex_test, zed_test = {}, {}

        for model_type in constants.model_types:
            if model_type=='esm-msa':
                msa_ind=True
            else:
                msa_ind=False
            
            ex_train[model_type], zed_train[model_type] = utils.pedestrian_input(list(df_cleared['brmid']), df_cleared, repr_path[model_type]['1325'], z_col='z-score', msa=msa_ind)
            # assemble the test data from the 117 set
            ex_test[model_type], zed_test[model_type] = utils.pedestrian_input(list(df_117['brmid']), df_117, repr_path[model_type]['117'], z_col='zscore', msa=msa_ind)

        # Quick check, whether the number of inputs is the same for all 3 model types 
        for model_type in constants.model_types:
            print(model_type)
            print('----------------------------')
            print('training set')
            print('input shape: ', ex_train[model_type].shape, 'output shape: ', zed_train[model_type].shape)
            print('test set')
            print('input shape: ', ex_test[model_type].shape, 'output shape: ', zed_test[model_type].shape)
            print()

        if ex_train[constants.model_types[0]].shape[0]==ex_train[constants.model_types[1]].shape[0]==ex_train[constants.model_types[2]].shape[0]:
            print('The number of inputs is the same for each model type')
        
        return ex_train, zed_train, ex_test, zed_test