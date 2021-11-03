import numpy as np
from sklearn.metrics import f1_score, adjusted_rand_score, accuracy_score, normalized_mutual_info_score
import pandas as pd
import sys
sys.path.append('..')


def cover_calculator(Y_pred):
    sample_num = Y_pred.shape[0]
    noise_num = Y_pred[Y_pred == -1].shape[0]
    return 1-noise_num/sample_num


def f1_score_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred != -1]
    Y_true_ = Y_true[Y_pred != -1]
    return f1_score(Y_true_, Y_pred_, average='weighted')


def ARI_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred != -1]
    Y_true_ = Y_true[Y_pred != -1]
    return adjusted_rand_score(Y_true_, Y_pred_)


def ACC_calculator(Y_true, Y_pred):
    Y_pred_ = Y_pred[Y_pred != -1]
    Y_true_ = Y_true[Y_pred != -1]
    return accuracy_score(Y_true_, Y_pred_)


def match(current, true_set, noise_set):
    max_overlap = 0
    idx = None
    for j in true_set.keys():
        N = len(current & true_set[j]-noise_set)
        if N > max_overlap:
            max_overlap = N
            idx = j
    return idx


def matchY(Y_pred, Y_true):
    if type(Y_true[0]) == str:
        select_mask = (Y_true != 'noise')
        Y_true = Y_true[select_mask]
        Y_true = Y_true.astype(np.int)
        Y_pred = Y_pred[select_mask]
    noise_mask = (Y_pred == -1)
    noise_set = set(np.nonzero(noise_mask)[0])

    # the set of predicted classes, for example：pred_set[-1]={816,501}
    pred_set = {}
    for i, val in enumerate(list(set(Y_pred))):
        pred_set[i] = set(np.where(Y_pred == val)[0])-noise_set

    true_set = {}
    for i, val in enumerate(sorted(list(set(Y_true)))):
        true_set[i] = set(np.where(Y_true == val)[0])

    Y_true = np.zeros_like(Y_pred)
    for i in true_set.keys():
        Y_true[list(true_set[i])] = i

    # sort the index of set according to its number of points
    sort_idx = np.argsort(-np.array(list(map(len, pred_set.values()))))

    # initial with -2, representing the false predicted points
    Y_pred = np.zeros_like(Y_pred)-2
    for i in sort_idx:
        if len(true_set) == 0:
            break
        pred_idx = list(pred_set.keys())[i]
        # find the real label of pred_set[pred_idx] from true_set
        real_y = match(pred_set[pred_idx], true_set, noise_set)
        if real_y is None:
            Y_pred[list(pred_set[pred_idx])] = -2
        else:
            Y_pred[list(pred_set[pred_idx])] = real_y
            del true_set[real_y]
    Y_pred[noise_mask] = -1

    return Y_pred, Y_true


def measures_calculator(Y_true, Y_pred):
    '''
    to calculation the measures for method evaluation;
    if multiclass is Ture, it will automatically find the best permutation of class label of Y_pred according to f1 score. 
    '''
    N_cls = len(set(Y_pred))
    if -1 in Y_pred:
        N_cls -= 1
    Y_pred, Y_true = matchY(Y_pred, Y_true)
    cover_rate = cover_calculator(Y_pred)
    Y_true = Y_true[Y_pred != -1]
    Y_pred = Y_pred[Y_pred != -1]
    df_mesures = pd.DataFrame(
        columns=['f1', 'ARI', 'ACC', 'NMI', 'cover_rate', 'classes'])
    # df_mesures = pd.DataFrame(columns=['f1', 'ARI', 'ACC', 'cover_rate'])

    f1 = f1_score_calculator(Y_true, Y_pred)
    ARI = ARI_calculator(Y_true, Y_pred)
    ACC = ACC_calculator(Y_true, Y_pred)
    NMI = normalized_mutual_info_score(Y_true, Y_pred)

    df_mesures.loc[0] = [f1, ARI, ACC, NMI, cover_rate, N_cls]
    # df_mesures.loc[0] = [f1, ARI, ACC, cover_rate]
    return df_mesures
