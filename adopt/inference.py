from joblib import load

def get_z_score(model_type: str = 'esm1b',
                sequence):
    reg = load('../models/lasso_'+model_type+'_cleared_sequence_cv.joblib')
    reg.predict(ex_test_seq)