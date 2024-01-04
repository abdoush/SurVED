from Data.dataset import Flchain, Nwtco, Metabric, Support
from rsfhelper import rsf_cv_nfolds
from Utils.helper import save_results
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

n_folds = 100
#2023-03-12 22:56:45,073:best_max_depth: 5, best_min_samples_split: 10, best_min_samples_leaf: 10, best_max_features: log2, best_c_index: 0.8383063589272594
n_estimators = 200
max_depth = 5
min_samples_split = 10
min_samples_leaf = 10
max_features = 'log2'
ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = rsf_cv_nfolds(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      max_features=max_features,
                      ds_class=Support, ds_file_name='support2',
                      final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_support, currentdir, n_folds=n_folds)


# 2023-03-12 23:40:38,380:best_max_depth: 15, best_min_samples_split: 15, best_min_samples_leaf: 15, best_max_features: sqrt, best_c_index: 0.7933822622297719
n_estimators = 200
max_depth = 15
min_samples_split = 15
min_samples_leaf = 15
max_features = 'sqrt'
ds_flchain = Flchain(f'{parentdir}/Data/flchain.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = rsf_cv_nfolds(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      max_features=max_features,
                      ds_class=Flchain, ds_file_name='flchain',
                      final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_flchain, currentdir, n_folds=n_folds)

#2023-03-12 23:48:47,184:best_max_depth: 5, best_min_samples_split: 10, best_min_samples_leaf: 100, best_max_features: log2, best_c_index: 0.7098614612263898
n_estimators = 200
max_depth = 5
min_samples_split = 10
min_samples_leaf = 100
max_features = 'log2'
ds_nwtco = Nwtco(f'{parentdir}/Data/nwtco.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = rsf_cv_nfolds(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      max_features=max_features,
                      ds_class=Nwtco, ds_file_name='nwtco',
                      final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_nwtco, currentdir, n_folds=n_folds)

#2023-03-12 23:59:18,064:best_max_depth: 15, best_min_samples_split: 25, best_min_samples_leaf: 5, best_max_features: sqrt, best_c_index: 0.6874359180105477
n_estimators = 200
max_depth = 15
min_samples_split = 25
min_samples_leaf = 5
max_features = 'sqrt'
ds_metabric = Metabric(f'{parentdir}/Data/metabric.csv', normalize_target=False, test_fract=0.3, number_of_splits=10)
cis, y_preds = rsf_cv_nfolds(n_estimators=n_estimators,
                      max_depth=max_depth,
                      min_samples_split=min_samples_split,
                      min_samples_leaf=min_samples_leaf,
                      max_features=max_features,
                      ds_class=Metabric, ds_file_name='metabric',
                      final_test=True, n_folds=n_folds)
print(cis)
save_results(y_preds, ds_metabric, currentdir, n_folds=n_folds)
