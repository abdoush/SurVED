
from Utils.metrics import c_index_decomposition
from lifelines.utils import concordance_index
import pandas as pd
from pycox.models import DeepHitSingle
import numpy as np
import torchtuples as tt
from Data.dataset import Support
import random
import torch
from Utils.helper import configure_logger
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


def deep_hit_fit(nc, alpha, sigma, ds, test_id=0, val_id=1, epochs=500, patience=100, batch_size=128, final_test=False):
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

    num_durations = nc  # int(np.max(y_train)/nc * 1.2)

    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda y, e: (y, e)
    yy_train = labtrans.fit_transform(*get_target(y_train, e_train))
    yy_val = labtrans.transform(*get_target(y_val, e_val))

    train = (x_train, yy_train)
    val = (x_val, yy_val)

    in_features = x_train.shape[1]
    # num_nodes = [32, 32]
    out_features = labtrans.out_features
    # batch_norm = True
    # dropout = 0.5

    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    net = get_net(in_features, out_features)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)
    model.optimizer.set_lr(1e-3)

    epochs = epochs
    batch_size = batch_size

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    log = model.fit(x_train, yy_train, batch_size, epochs, callbacks, verbose=0, val_data=val)

    surv_test = model.predict_surv_df(x_test)

    c_index = concordance_index(y_test, np.sum(surv_test), e_test)
    return c_index, np.sum(surv_test)


def deep_hit_cv(nc, alpha, sigma, ds, epochs=500, patience=100, batch_size=128, final_test=False):
    cis = []
    y_preds = []
    for val_id, test_id in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]:
        ci, y_pred = deep_hit_fit(nc, alpha, sigma, ds, test_id, val_id, epochs, patience, batch_size,
                                  final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)

    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)

def deep_hit_cv_nfolds(nc, alpha, sigma, ds_class, ds_file_name, epochs=500, patience=100, batch_size=128, final_test=False, n_folds=100):
    cis = []
    y_preds = []
    for val_id in range(n_folds):
        ds = ds_class(f'{parentdir}/Data/{ds_file_name}.csv', normalize_target=False, test_fract=0.3, number_of_splits=10, train_splits_seed=val_id)
        ci, y_pred = deep_hit_fit(nc, alpha, sigma, ds, 0, 0, epochs, patience, batch_size, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)
        print(f'{val_id}: {ci}')
    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def random_search(ds, epochs=500, patience=100, batch_size=128, no_change=10, logdir=None):
    # (x_train, ye_train, y_train, e_train,
    #  x_val, ye_val, y_val, e_val,
    #  x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=0, val_id=1)
    #
    # x_train = x_train.astype('float32')
    # x_val = x_val.astype('float32')

    logger = configure_logger(ds, logdir)

    selected = []

    nc = 100
    alpha = 0.01
    sigma = 1

    best_c_index = 0
    counter = 0

    while (counter < no_change):
        if (alpha, sigma) not in selected:
            logger.info(f'Counter: {counter}, Testing alpha: {alpha}, sigma: {sigma}, nc: {nc}')
            selected.append((alpha, sigma))
            counter += 1

            c_index = deep_hit_cv(nc=nc, alpha=alpha, sigma=sigma, ds=ds, epochs=epochs, patience=patience,
                                  batch_size=batch_size)

            logger.info(c_index)

            if (c_index > best_c_index):
                counter = 0
                best_c_index = c_index
                logger.info(f'New best c-index: {c_index}')
                logger.info('=================================================================')
                best_alpha = alpha
                best_sigma = sigma
                best_nc = nc

        nc = random.choice([100, 200, 400, 1000])
        alpha = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9,
                               1])  # [0.001, 0.01, 0.05, 0.1, 0.5, 0.8 , 0.9, 1])
        sigma = random.choice([0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9,
                               1])  # [0.001, 0.01, 0.05, 0.1, 0.5, 0.8 , 0.9, 1])

    return best_alpha, best_sigma, best_nc

def deep_hit_fit_change_censoring(nc, alpha, sigma, ds, test_id=0, val_id=1, epochs=500, patience=100, batch_size=128, final_test=False):
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

    num_durations = nc  # int(np.max(y_train)/nc * 1.2)

    labtrans = DeepHitSingle.label_transform(num_durations)
    get_target = lambda y, e: (y, e)
    yy_train = labtrans.fit_transform(*get_target(y_train, e_train))
    yy_val = labtrans.transform(*get_target(y_val, e_val))

    train = (x_train, yy_train)
    val = (x_val, yy_val)

    in_features = x_train.shape[1]
    # num_nodes = [32, 32]
    out_features = labtrans.out_features
    # batch_norm = True
    # dropout = 0.5

    # net = tt.practical.MLPVanilla(in_features, num_nodes, out_features, batch_norm, dropout)
    net = get_net(in_features, out_features)
    model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha, sigma=sigma, duration_index=labtrans.cuts)
    model.optimizer.set_lr(1e-3)

    epochs = epochs
    batch_size = batch_size

    callbacks = [tt.callbacks.EarlyStopping(patience=patience)]
    log = model.fit(x_train, yy_train, batch_size, epochs, callbacks, verbose=0, val_data=val)

    surv_test = model.predict_surv_df(x_test)

    c_index = concordance_index(y_test, np.sum(surv_test), e_test)
    return c_index, np.sum(surv_test), y_test, e_test

def deephit_change_size_only(nc, alpha, sigma, epochs=500, patience=100, batch_size=128, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.60, 0.51, 0.36, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='drop', events_only=False)
        ci, y_pred, y_test, e_test = deep_hit_fit_change_censoring(nc=nc, alpha=alpha, sigma=sigma, ds=ds_support, test_id=1, val_id=0, epochs=epochs, patience=patience, batch_size=batch_size, final_test=final_test)
        #deep_hit_fit_change_censoring(ds=ds_support,reg=reg, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/deephit_final_results_change_size_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

def deephit_change_censoring_only(nc, alpha, sigma, epochs=500, patience=100, batch_size=128, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='censor', events_only=True)
        ci, y_pred, y_test, e_test = deep_hit_fit_change_censoring(nc=nc, alpha=alpha, sigma=sigma, ds=ds_support, test_id=1, val_id=0, epochs=epochs, patience=patience, batch_size=batch_size, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/deephit_final_results_change_censoring_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

def deephit_change_censoring_and_size(nc, alpha, sigma, epochs=500, patience=100, batch_size=128, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='drop', events_only=True)
        ci, y_pred, y_test, e_test = deep_hit_fit_change_censoring(nc=nc, alpha=alpha, sigma=sigma, ds=ds_support, test_id=1, val_id=0, epochs=epochs, patience=patience, batch_size=batch_size, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/deephit_final_results_change_censoring_and_size_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)
