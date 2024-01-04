from Data.dataset import Flchain, Nwtco, Metabric, Support
from cphhelper import cph_cv_nfolds
from Utils.helper import save_results
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

n_folds = 100
ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=True, test_fract=0.3, number_of_splits=10)
cis, y_preds = cph_cv_nfolds(ds_class=Support, ds_file_name='support2', reg=0.1, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_support, currentdir, n_folds=n_folds)

ds_flchain = Flchain(f'{parentdir}/Data/flchain.csv', normalize_target=True, test_fract=0.3, number_of_splits=10)
cis, y_preds = cph_cv_nfolds(ds_class=Flchain, ds_file_name='flchain', reg=0.1, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_flchain, currentdir, n_folds=n_folds)

ds_nwtco = Nwtco(f'{parentdir}/Data/nwtco.csv', normalize_target=True, test_fract=0.3, number_of_splits=10)
cis, y_preds = cph_cv_nfolds(ds_class=Nwtco, ds_file_name='nwtco', reg=0.1, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_nwtco, currentdir, n_folds=n_folds)

ds_metabric = Metabric(f'{parentdir}/Data/metabric.csv', normalize_target=True, test_fract=0.3, number_of_splits=10)
cis, y_preds = cph_cv_nfolds(ds_class=Metabric, ds_file_name='metabric', reg=0.1, final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_metabric, currentdir, n_folds=n_folds)