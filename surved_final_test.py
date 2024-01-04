from survedhelper import surved_fit
from Utils.helper import save_results
from Model.experiments import FlchainExperiment, MetabricExperiment, NwtcoExperiment, SupportExperiment
from Data.dataset import Flchain, Nwtco, Metabric, Support

import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

n_folds = 100
epochs = 10000000
patience = 1000

# Weights:
# Testing events_weight: 0.001, censored_weight: 0.8, c_index_lb_weight: 0.8, kl_loss_weight: 0.0005
events_weight = 0.001
censored_weight = 0.8
c_index_lb_weight = 0.8
kl_loss_weight = 0.0005
ds_support = Support('Data/support2.csv', test_fract=0.3)
ci_tests, y_test_preds = surved_fit(exp_model=SupportExperiment, i='Final_Test',
                                   events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                                   epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(ci_tests)
save_results(y_test_preds, ds_support, currentdir, n_folds=n_folds)


# Weights:
# Testing events_weight: 0.001, censored_weight: 0.005, c_index_lb_weight: 0.05, kl_loss_weight: 5e-05
events_weight = 0.001
censored_weight = 0.005
c_index_lb_weight = 0.05
kl_loss_weight = 5e-05
ds_flchain = Flchain('Data/flchain.csv', test_fract=0.3)
ci_tests, y_test_preds = surved_fit(exp_model=FlchainExperiment, i='Final_Test',
                                   events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                                   epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(ci_tests)
save_results(y_test_preds, ds_flchain, currentdir, n_folds=n_folds)


# Weights:
# Testing events_weight: 0.0001, censored_weight: 0.1, c_index_lb_weight: 0.5, kl_loss_weight: 0.0001
events_weight = 0.0001
censored_weight = 0.1
c_index_lb_weight = 0.5
kl_loss_weight = 0.0001
ds_nwtco = Nwtco('Data/nwtco.csv', test_fract=0.3)
ci_tests, y_test_preds = surved_fit(exp_model=NwtcoExperiment, i='Final_Test',
                                   events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                                   epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(ci_tests)
save_results(y_test_preds, ds_nwtco, currentdir, n_folds=n_folds)


# Weights:
# Testing events_weight: 0.0001, censored_weight: 0.0001, c_index_lb_weight: 0.1, kl_loss_weight: 0.05
events_weight = 0.0001
censored_weight = 0.0001
c_index_lb_weight = 0.1
kl_loss_weight = 0.05
ds_metabric = Metabric('Data/metabric.csv', test_fract=0.3)
ci_tests, y_test_preds = surved_fit(exp_model=MetabricExperiment, i='Final_Test',
                                   events_weight=events_weight, censored_weight=censored_weight, kl_loss_weight=kl_loss_weight, c_index_lb_weight=c_index_lb_weight,
                                   epochs=epochs, patience=patience, batch_size=256, final_test=True, n_folds=n_folds)
print(ci_tests)
save_results(y_test_preds, ds_metabric, currentdir, n_folds=n_folds)

