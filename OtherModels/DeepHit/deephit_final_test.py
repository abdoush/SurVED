from Data.dataset import Flchain, Nwtco, Metabric, Support
from deephithelper import deep_hit_cv_nfolds
from Utils.helper import save_results
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

n_folds = 100
epochs = 10000000
patience = 1000

# 2023-03-03 02:06:47,711:Counter: 8, Testing alpha: 0.05, sigma: 0.8, nc: 400
# 2023-03-03 02:16:33,227:0.8706224447532058
nc = 400
alpha = 0.05
sigma = 0.8
ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = deep_hit_cv_nfolds(nc=nc, alpha=alpha, sigma=sigma, ds_class=Support, ds_file_name='support2', epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_support, currentdir, n_folds=n_folds)


# 2023-03-03 16:28:48,026:Counter: 12, Testing alpha: 0.05, sigma: 0.5, nc: 200
# 2023-03-03 16:34:15,619:0.8000223214386303
nc = 200
alpha = 0.05
sigma = 0.5
ds_flchain = Flchain(f'{parentdir}/Data/flchain.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = deep_hit_cv_nfolds(nc=nc, alpha=alpha, sigma=sigma, ds_class=Flchain, ds_file_name='flchain', epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_flchain, currentdir, n_folds=n_folds)

# 2023-03-04 02:32:00,044:Counter: 22, Testing alpha: 0.5, sigma: 1, nc: 200
# 2023-03-04 02:34:42,724:0.7064049764704116
nc = 200
alpha = 0.5
sigma = 1
ds_nwtco = Nwtco(f'{parentdir}/Data/nwtco.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = deep_hit_cv_nfolds(nc=nc, alpha=alpha, sigma=sigma, ds_class=Nwtco, ds_file_name='nwtco', epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_nwtco, currentdir, n_folds=n_folds)

# 2023-03-04 06:41:42,199:Counter: 10, Testing alpha: 0.9, sigma: 1, nc: 100
# 2023-03-04 06:42:56,276:0.6476107732600469
nc = 100
alpha = 0.9
sigma = 1
ds_metabric = Metabric(f'{parentdir}/Data/metabric.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = deep_hit_cv_nfolds(nc=nc, alpha=alpha, sigma=sigma, ds_class=Metabric, ds_file_name='metabric', epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_metabric, currentdir, n_folds=n_folds)

