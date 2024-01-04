import logging
import pandas as pd

def configure_logger(exp_model, logdir):
    name = exp_model.__name__
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s:%(message)s')
    file_handler = logging.FileHandler(logdir + '/' + name + '.log')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def save_results(y_preds, ds, resdir, n_folds=100):
    df = pd.DataFrame(y_preds.transpose(), columns=[f'F{i}' for i in range(n_folds)])
    (x_train, ye_train, y_train, e_train,
     x_val, ye_val, y_val, e_val,
     x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=0)
    df['T'] = y_test
    df['E'] = e_test
    df.to_csv(f'{resdir}/final_results_{ds.get_dataset_name()}.csv', index=False)