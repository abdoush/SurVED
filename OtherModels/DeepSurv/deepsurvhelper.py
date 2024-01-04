from Data.dataset import Flchain, Support, Nwtco, Metabric
from lifelines.utils import concordance_index
import pandas as pd
from pycox.models import CoxPH

import numpy as np
import torchtuples as tt
import torch
import sys, os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def get_net(in_features, out_features):
    net = torch.nn.Sequential(
        torch.nn.Linear(in_features, 32),
        torch.nn.Tanh(),
        # torch.nn.BatchNorm1d(32),
        torch.nn.Dropout(0.5),

        torch.nn.Linear(32, 32),
        torch.nn.Tanh(),
        # torch.nn.Dropout(0.1),

        torch.nn.Linear(32, out_features)
    )
    return net


def deep_surv_fit(ds, test_id=0, val_id=1, epochs=500, patience=100, batch_size=128, final_test=False):
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    torch.manual_seed(20)

    get_target = lambda y ,e: (y, e)
    yy_train = get_target(y_train, e_train)
    yy_val = get_target(y_val, e_val)

    val = (x_val, yy_val)

    in_features = x_train.shape[1]
    out_features = 1

    net = get_net(in_features, out_features)

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(1e-3)

    epochs = epochs
    batch_size = batch_size

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    log = model.fit(x_train, yy_train, batch_size, epochs, callbacks, verbose=0, val_data=val)
    model.compute_baseline_hazards()
    surv_test = model.predict_surv_df(x_test)
    surv_val = model.predict_surv_df(x_val)
    surv_train = model.predict_surv_df(x_train)
    c_index_train = concordance_index(y_train, np.sum(surv_train), e_train)
    c_index_val = concordance_index(y_val, np.sum(surv_val), e_val)

    c_index = concordance_index(y_test, np.sum(surv_test), e_test)
    print('train, val, test:', c_index_train, c_index_val, c_index)
    return c_index, np.sum(surv_test)


def deep_surv_cv(ds, epochs=500, patience=100, batch_size=128, final_test=False):
    cis = []
    y_preds = []
    for val_id, test_id in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]:
        ci, y_pred = deep_surv_fit(ds, test_id, val_id, epochs, patience, batch_size, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)

    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def deep_hit_cv_nfolds(ds_class, ds_file_name, epochs=500, patience=100, batch_size=128, final_test=False, n_folds=100):
    cis = []
    y_preds = []
    for val_id in range(n_folds):
        ds = ds_class(f'{parentdir}/Data/{ds_file_name}.csv', normalize_target=True, test_fract=0.3, number_of_splits=10, train_splits_seed=val_id)
        ci, y_pred = deep_surv_fit(ds, 0, 0, epochs, patience, batch_size, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)
        print(f'{val_id}: {ci}')
    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def deep_surv_fit_change_censoring(ds, test_id=0, val_id=1, epochs=500, patience=100, batch_size=128, final_test=False):
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    torch.manual_seed(20)

    get_target = lambda y ,e: (y, e)
    yy_train = get_target(y_train, e_train)
    yy_val = get_target(y_val, e_val)

    val = (x_val, yy_val)

    in_features = x_train.shape[1]
    out_features = 1 #labtrans.out_features

    net = get_net(in_features, out_features)

    #num_nodes = [64, 64, 64, 64]
    # out_features = labtrans.out_features
    # batch_norm = True
    # dropout = 0.5
    # torch.manual_seed(20)
    # net = tt.practical.MLPVanilla(in_features, num_nodes, 1, batch_norm, dropout, output_bias=False)

    model = CoxPH(net, tt.optim.Adam)
    model.optimizer.set_lr(1e-3)

    epochs = epochs
    batch_size = batch_size #x_train.shape[0]//4

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    log = model.fit(x_train, yy_train, batch_size, epochs, callbacks, verbose=0, val_data=val)
    model.compute_baseline_hazards()
    surv_test = model.predict_surv_df(x_test)
    surv_val = model.predict_surv_df(x_val)
    surv_train = model.predict_surv_df(x_train)
    c_index_train = concordance_index(y_train, np.sum(surv_train), e_train)
    c_index_val = concordance_index(y_val, np.sum(surv_val), e_val)

    c_index = concordance_index(y_test, np.sum(surv_test), e_test)
    print('train, val, test:', c_index_train, c_index_val, c_index)
    return c_index, np.sum(surv_test), y_test, e_test


def deep_surv_change_size_only(epochs=500, patience=100, batch_size=128, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.60, 0.51, 0.36, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, p=pe, action='drop', events_only=False)
        ci, y_pred, y_test, e_test = deep_surv_fit_change_censoring(ds=ds_support, test_id=1, val_id=0, epochs=epochs, patience=patience, batch_size=batch_size, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/deep_surv_final_results_change_size_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)


def deep_surv_change_censoring_only(epochs=500, patience=100, batch_size=128, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, p=pe, action='censor', events_only=True)
        ci, y_pred, y_test, e_test = deep_surv_fit_change_censoring(ds=ds_support, test_id=1, val_id=0, epochs=epochs, patience=patience, batch_size=batch_size, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/deep_surv_final_results_change_censoring_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)


def deep_surv_change_censoring_and_size(epochs=500, patience=100, batch_size=128, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, p=pe, action='drop', events_only=True)
        ci, y_pred, y_test, e_test = deep_surv_fit_change_censoring(ds=ds_support, test_id=1, val_id=0, epochs=epochs, patience=patience, batch_size=batch_size, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/deep_surv_final_results_change_censoring_and_size_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

