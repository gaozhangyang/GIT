'''
measures for results
'''

import numpy as np
from sklearn.metrics import f1_score, adjusted_rand_score, accuracy_score
from sympy.utilities.iterables import multiset_permutations
import pandas as pd
def measures_calculator(Y_true, Y_pred, multiclass=False):
    '''
    to calculation the measures for method evaluation;
    if multiclass is Ture, it will automatically find the best permutation of class label of Y_pred according to f1 score. 
    '''
    cover_rate = cover_calculator(Y_pred)
    Y_true = Y_true[Y_pred!=-1]
    Y_pred = Y_pred[Y_pred!=-1]
    df_mesures = pd.DataFrame(columns=['f1', 'ARI', 'ACC', 'cover_rate'])
    if multiclass == True:
        _, Y_pred = multi_class_permutation(Y_true, Y_pred)

    f1 = f1_score_calculator(Y_true, Y_pred)
    ARI = ARI_calculator(Y_true, Y_pred)
    ACC = ACC_calculator(Y_true, Y_pred)

    df_mesures.loc[0] = [f1, ARI, ACC, cover_rate]
    return df_mesures

def cover_calculator(Y_pred):
    sample_num = Y_pred.shape[0]
    noise_num = Y_pred[Y_pred==-1].shape[0]
    return 1-noise_num/sample_num


def f1_score_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred!=-1]
    Y_true_ = Y_true[Y_pred!=-1]
    return f1_score(Y_true_, Y_pred_, average='weighted')

def ARI_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred!=-1]
    Y_true_ = Y_true[Y_pred!=-1]
    return adjusted_rand_score(Y_true_, Y_pred_)

def ACC_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred!=-1]
    Y_true_ = Y_true[Y_pred!=-1]
    return accuracy_score(Y_true_, Y_pred_)

def multi_class_permutation(Y_true, Y_pred):
    classes = np.unique(Y_true)
    class_num = len(classes)
    Y_pred_ = Y_pred[Y_pred!=-1]
    Y_true_ = Y_true[Y_pred!=-1]

    f1_best = 0
    Y_pred_best = Y_pred_
    for p in multiset_permutations(classes):
        mapping_dict = {classes[i]:p[i] for i in range(class_num)}
        def mapping(x):
            return mapping_dict[x]
        vec_mapping=np.vectorize(mapping)
        Y_pred_current = vec_mapping(Y_pred_)
        f1 = f1_score(Y_true_, Y_pred_current, average='weighted')
        if f1_best <= f1:
            f1_best = f1
            Y_pred_best = np.copy(Y_pred_current)
    
    return f1_best, Y_pred_best


import numpy as np
if __name__ == '__main__':
    Y_true = np.array([1,2,3,1,2,3,1,2,3])
    Y_pred = np.array([-1,2,3,1,3,2,1,1,1])

    print(measures_calculator(Y_true, Y_pred, multiclass=True))

