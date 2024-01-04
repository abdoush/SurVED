from Model.experiments import FlchainExperiment, MetabricExperiment, NwtcoExperiment, SupportExperiment
from lifelines.utils import concordance_index
import random
import gc
import pandas as pd
import numpy as np
import sys, os, inspect
from Utils.helper import configure_logger

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def surved_fit(exp_model, i, events_weight=1, censored_weight=1, kl_loss_weight=0.0001, c_index_lb_weight=0, epochs=10000, patience=1000, batch_size=256, final_test=False, n_folds=100):
    LR = 0.001

    if final_test:
        VAL_IDS_LIST = range(n_folds)
        exp = exp_model(exp_name=f'SurVED_Final_Test_{i}',
                        events_weight=events_weight, censored_weight=censored_weight,
                        surv_mse_loss_weight=1, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                        max_epochs=epochs, patience=patience, surved_lr=LR, batch_size=batch_size,
                        latent_size=4, activation='tanh',
                        num_samples=100, verbose=False
                        )
        ci_test, ci_val, ci_train, ci_tests, ci_vals, ci_trains, y_test_preds = exp.run_final_test(val_ids_lst=VAL_IDS_LIST)
    else:
        TEST_VAL_IDS_LIST = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        exp = exp_model(exp_name=f'SurVED_{i}',
                        events_weight=events_weight, censored_weight=censored_weight,
                        surv_mse_loss_weight=1, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                        max_epochs=epochs, patience=patience, surved_lr=LR, batch_size=batch_size,
                        latent_size=4, activation='tanh',
                        num_samples=100, verbose=False
                        )
        ci_test, ci_val, ci_train, ci_tests, ci_vals, ci_trains, y_test_preds = exp.run_cv(test_val_ids_lst=TEST_VAL_IDS_LIST)
    del exp
    gc.collect()
    return ci_test, y_test_preds


def random_search(exp_model,df, logdir, epochs=10000, patience=1000, batch_size=256, no_change=10):
    logger = configure_logger(exp_model, logdir)

    selected = []
    events_weight = 0.0001
    censored_weight = 0.1
    c_index_lb_weight = 0.1
    kl_loss_weight = 0.05
    best_c_index = 0
    best_events_weight = events_weight
    best_censored_weight = censored_weight
    best_c_index_lb_weight = c_index_lb_weight
    best_kl_loss_weight = kl_loss_weight
    best_i = 0
    counter = 0
    i = 0
    num_selected = 0
    while ((counter < no_change) and (num_selected < 100)):
        i += 1
        logger.info(f'{i} - Testing events_weight: {events_weight}, censored_weight: {censored_weight}, c_index_lb_weight: {c_index_lb_weight}, kl_loss_weight: {kl_loss_weight}')

        if (events_weight, censored_weight, c_index_lb_weight, kl_loss_weight) not in list(
                map(tuple, df.iloc[:, 1:-1].values)):  # selected:
            selected.append((events_weight, censored_weight, c_index_lb_weight, kl_loss_weight))
            counter += 1

            c_index, y_test_preds = surved_fit(exp_model=exp_model, i=i,
                                 events_weight=events_weight, censored_weight=censored_weight,
                                 kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                                 epochs=epochs, patience=patience, batch_size=batch_size
                                 )

            logger.info(c_index)
            df.loc[len(df)] = [i, events_weight, censored_weight, c_index_lb_weight, kl_loss_weight, c_index]
            df.to_csv(f'{exp_model.__name__}_results.csv', index=False)
            if (c_index > best_c_index):
                counter = 0
                num_selected = 0
                best_c_index = c_index
                best_i = i
                logger.info(f'New best c-index: {str(c_index)}')
                logger.info('=================================================================')
                best_events_weight = events_weight
                best_censored_weight = censored_weight
                best_c_index_lb_weight = c_index_lb_weight
                best_kl_loss_weight = kl_loss_weight
        else:
            #print('Already Selected')
            num_selected += 1
        # random.seed(i)
        events_weight = random.choice([0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9, 1])
        censored_weight = random.choice([0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9, 1])
        c_index_lb_weight = random.choice([0, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 0.8, 0.9, 1])
        kl_loss_weight = random.choice([0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

    return best_events_weight, best_censored_weight, best_c_index_lb_weight, best_kl_loss_weight, best_i, best_c_index


def surved_fit_change_censoring(i, pe, events_only, action,  events_weight=1, censored_weight=1, kl_loss_weight=0.0001, c_index_lb_weight=0, epochs=10000, patience=1000, batch_size=256):
    LR = 0.001
    exp_support = SupportExperiment(exp_name=f'SurVED_Final_Test_{i}',
                                    drop_percentage=pe, events_only=events_only, action=action,
                                    events_weight=events_weight, censored_weight=censored_weight,
                                    surv_mse_loss_weight=1, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                                    max_epochs=epochs, patience=patience, surved_lr=LR, batch_size=batch_size,
                                    latent_size=4, activation='tanh',
                                    num_samples=100, verbose=False
                                    )
    model = exp_support.run_fold(test_id=None, val_id=0, fold_id=i, is_tuning=False)
    c_index = concordance_index(model.y_test, model.y_test_pred, model.e_test)
    return c_index, model.y_test_pred, model.y_test, model.e_test


def surved_change_size_only(events_weight=1, censored_weight=1, kl_loss_weight=0.0001, c_index_lb_weight=0, epochs=10000, patience=1000, batch_size=256):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for i, pe in enumerate([0.60, 0.51, 0.36, 'full']):
        ci, y_pred, y_test, e_test = surved_fit_change_censoring(i=i, pe=pe, action='drop', events_only=False, events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight, epochs=epochs, patience=patience, batch_size=batch_size)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/surved_final_results_change_size_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)


def surved_change_censoring_only(events_weight=1, censored_weight=1, kl_loss_weight=0.0001, c_index_lb_weight=0, epochs=10000, patience=1000, batch_size=256):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for i, pe in enumerate([0.20, 0.35, 0.50, 'full']):
        ci, y_pred, y_test, e_test = surved_fit_change_censoring(i=i, pe=pe, action='censor', events_only=True, events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight, epochs=epochs, patience=patience, batch_size=batch_size)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/surved_final_results_change_censoring_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)


def surved_change_censoring_and_size(events_weight=1, censored_weight=1, kl_loss_weight=0.0001, c_index_lb_weight=0, epochs=10000, patience=1000, batch_size=256):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for i, pe in enumerate([0.20, 0.35, 0.50, 'full']):
        ci, y_pred, y_test, e_test = surved_fit_change_censoring(i=i, pe=pe, action='drop', events_only=True, events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight, epochs=epochs, patience=patience, batch_size=batch_size)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/surved_final_results_change_censoring_and_size_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

