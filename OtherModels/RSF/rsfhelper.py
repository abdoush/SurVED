from sksurv.ensemble import RandomSurvivalForest
import numpy as np
from Utils.helper import configure_logger
import random
import pandas as pd
from Data.dataset import Support
import sys, os, inspect
from lifelines.utils import concordance_index
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def rsf_fit(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, ds, test_id=0, val_id=1,
            final_test=False):
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    dt = np.dtype('bool,float')
    y_train_surv = np.array([(bool(e), y) for e, y in zip(e_train, y_train)], dtype=dt)
    y_val_surv = np.array([(bool(e), y) for e, y in zip(e_val, y_val)], dtype=dt)
    y_test_surv = np.array([(bool(e), y) for e, y in zip(e_test, y_test)], dtype=dt)

    # train RSF
    rsf = RandomSurvivalForest(n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               max_features=max_features,
                               oob_score=True,
                               n_jobs=-1,
                               random_state=20)
    rsf.fit(x_train, y_train_surv)

    #c_index = rsf.score(x_test, y_test_surv)

    surv_test = -rsf.predict(x_test)
    c_index = concordance_index(y_test, surv_test, e_test)

    return c_index, surv_test


def rsf_cv(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, ds, final_test=False):
    cis = []
    y_preds = []
    for val_id, test_id in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]:
        ci, y_pred = rsf_fit(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, max_features=max_features, ds=ds, test_id=test_id,
                             val_id=val_id, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)

    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)

def rsf_cv_nfolds(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, ds_class, ds_file_name, final_test=False, n_folds=100):
    cis = []
    y_preds = []
    for val_id in range(n_folds):
        ds = ds_class(f'{parentdir}/Data/{ds_file_name}.csv', normalize_target=False, test_fract=0.3, number_of_splits=10, train_splits_seed=val_id)
        ci, y_pred = rsf_fit(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, max_features=max_features, ds=ds, test_id=0,
                             val_id=0, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)
        print(f'{val_id}: {ci}')
    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def random_search(ds, logdir, no_change=10):
    logger = configure_logger(ds, logdir)
    selected = []

    max_depth = 5
    min_samples_split = 10
    min_samples_leaf = 5
    max_features = 'log2'

    best_c_index = 0
    counter = 0

    while (counter < no_change):

        if (max_depth, min_samples_split, min_samples_leaf, max_features) not in selected:
            logger.info(
                f'Testing max_depth: {max_depth}, min_samples_split: {min_samples_split}, min_samples_leaf: {min_samples_leaf}, max_features: {max_features}')
            selected.append((max_depth, min_samples_split, min_samples_leaf, max_features))
            counter += 1

            c_index = rsf_cv(n_estimators=200, max_depth=max_depth, min_samples_split=min_samples_split,
                             min_samples_leaf=min_samples_leaf, max_features=max_features, ds=ds)

            logger.info(c_index)

            if (c_index > best_c_index):
                counter = 0
                best_c_index = c_index
                logger.info(f'New best c-index: {c_index}')
                logger.info('=================================================================')
                best_max_depth = max_depth
                best_min_samples_split = min_samples_split
                best_min_samples_leaf = min_samples_leaf
                best_max_features = max_features

        max_depth = random.choice([5, 10, 15, 20, None])
        min_samples_split = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 100])
        min_samples_leaf = random.choice([5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 100])
        max_features = random.choice(['log2', 'sqrt'])

    logger.info(
        f'best_max_depth: {best_max_depth}, best_min_samples_split: {best_min_samples_split}, best_min_samples_leaf: {best_min_samples_leaf}, best_max_features: {best_max_features}, best_c_index: {best_c_index}')
    return best_max_depth, best_min_samples_split, best_min_samples_leaf, best_max_features, best_c_index


def rsf_fit_change_censoring(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, ds, test_id=0, val_id=1, final_test=False):
    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    dt = np.dtype('bool,float')
    y_train_surv = np.array([(bool(e), y) for e, y in zip(e_train, y_train)], dtype=dt)
    y_val_surv = np.array([(bool(e), y) for e, y in zip(e_val, y_val)], dtype=dt)
    y_test_surv = np.array([(bool(e), y) for e, y in zip(e_test, y_test)], dtype=dt)

    # train RSF
    rsf = RandomSurvivalForest(n_estimators=n_estimators,
                               max_depth=max_depth,
                               min_samples_split=min_samples_split,
                               min_samples_leaf=min_samples_leaf,
                               max_features=max_features,
                               oob_score=True,
                               n_jobs=-1,
                               random_state=20)
    rsf.fit(x_train, y_train_surv)

    c_index = rsf.score(x_test, y_test_surv)

    surv_test = -rsf.predict(x_test)

    return c_index, surv_test, y_test, e_test


def rsf_change_size_only(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.60, 0.51, 0.36, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='drop', events_only=False)
        ci, y_pred, y_test, e_test = rsf_fit_change_censoring(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, ds=ds_support, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/rsf_final_results_change_size_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

def rsf_change_censoring_only(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='censor', events_only=True)
        ci, y_pred, y_test, e_test = rsf_fit_change_censoring(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, ds=ds_support, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/rsf_final_results_change_censoring_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

def rsf_change_censoring_and_size(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='drop', events_only=True)
        ci, y_pred, y_test, e_test = rsf_fit_change_censoring(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, max_features=max_features, ds=ds_support, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/rsf_final_results_change_censoring_and_size_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)
