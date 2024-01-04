from Utils.helper import configure_logger
import numpy as np
from lifelines.utils import concordance_index
from sksurv.linear_model import CoxPHSurvivalAnalysis
import pandas as pd
from Data.dataset import Support
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def cph_fit(ds, reg=0.1, test_id=0, val_id=1, final_test=False):
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

    cph = CoxPHSurvivalAnalysis(alpha=reg).fit(x_train, y_train_surv)
    surv_test = -cph.predict(x_test)
    c_index = concordance_index(y_test, surv_test, e_test)

    return c_index, surv_test

def cph_cv(ds, reg, final_test=False):
    cis = []
    y_preds = []
    for val_id, test_id in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]:
        ci, y_pred = cph_fit(ds=ds,reg=reg, test_id=test_id, val_id=val_id, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)

    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def cph_cv_nfolds(ds_class, ds_file_name, reg, final_test=False, n_folds=100):
    cis = []
    y_preds = []
    for val_id in range(n_folds):
        ds = ds_class(f'{parentdir}/Data/{ds_file_name}.csv', normalize_target=True, test_fract=0.3, number_of_splits=10, train_splits_seed=val_id)
        ci, y_pred = cph_fit(ds=ds,reg=reg, test_id=0, val_id=0, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)
        print(f'{val_id}: {ci}')
    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def cph_fit_change_censoring(ds, reg=0.1, test_id=0, val_id=1, final_test=False):
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

    cph = CoxPHSurvivalAnalysis(alpha=reg).fit(x_train, y_train_surv)
    surv_test = -cph.predict(x_test)
    c_index = concordance_index(y_test, surv_test, e_test)

    return c_index, surv_test, y_test, e_test


def cph_change_size_only(reg, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.60, 0.51, 0.36, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, p=pe, action='drop', events_only=False)
        ci, y_pred, y_test, e_test = cph_fit_change_censoring(ds=ds_support,reg=reg, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/cph_final_results_change_size_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

def cph_change_censoring_only(reg, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, p=pe, action='censor', events_only=True)
        ci, y_pred, y_test, e_test = cph_fit_change_censoring(ds=ds_support,reg=reg, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/cph_final_results_change_censoring_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

def cph_change_censoring_and_size(reg, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, p=pe, action='drop', events_only=True)
        ci, y_pred, y_test, e_test = cph_fit_change_censoring(ds=ds_support,reg=reg, test_id=1, val_id=0, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/cph_final_results_change_censoring_and_size_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)
