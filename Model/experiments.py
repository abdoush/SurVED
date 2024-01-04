from Model.model import SurVED
import numpy as np
from Data.dataset import Metabric, Flchain, Nwtco, Support
import os
import gc

class Fold:
    def __init__(self, base_dir, fold_id, dataset, structure,
                 events_weight, censored_weight, surv_mse_loss_weight, kl_loss_weight, c_index_lb_weight,
                 surved_lr=0.00001, is_tuning=True,
                 surv_regression_error_func='abs',
                 patience=10000, max_epochs=1000000, verbose=True,
                 surved_monitor='val_cindex', surved_mode='max',
                 activate_tensorboard=False, batch_size=256, activation='sigmoid',
                 val_id=0, test_id=1,
                 num_samples=100,
                 pretrained_file_path=None
                 ):
        self.activation = activation
        self.dataset = dataset
        self.structure = structure
        self.base_dir = base_dir
        self.fold_id = fold_id
        self.events_weight = events_weight
        self.censored_weight = censored_weight
        self.surv_mse_loss_weight = surv_mse_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.c_index_lb_weight = c_index_lb_weight
        self.surved_lr = surved_lr
        self.is_tuning = is_tuning
        self.surv_regression_error_func = surv_regression_error_func
        self.patience = patience
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.surved_monitor = surved_monitor
        self.surved_mode = surved_mode
        self.activate_tensorboard = activate_tensorboard
        self.batch_size = batch_size
        self.test_id = test_id
        self.val_id = val_id
        self.num_samples = num_samples
        self.pretrained_file_path = pretrained_file_path

    def _make_fold_folder(self, val_id, test_id, fold_id):
        if test_id is None:
            test_id = '_testset'
        self.folder_name = self.base_dir + 'Exp{}_Val{}_Test{}_{}_Results/'.format(fold_id, val_id, test_id, self.dataset.get_dataset_name())
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)

    def fit(self):
        encoder_structure = self.structure['encoder']
        decoder_y_structure = self.structure['decoder_y']

        self._make_fold_folder(val_id=self.val_id, test_id=self.test_id, fold_id=self.fold_id)
        self.model_file_name = "SurVED_best.h5"
        self.log_file_name = 'Experiment.log'
        self.surv_model = SurVED(input_shape=self.dataset.df.shape[1] - 2,
                            encoder_structure_dict=encoder_structure,
                            decoder_y_structure_dict=decoder_y_structure,
                            activation=self.activation,
                            events_weight=self.events_weight,
                            censored_weight=self.censored_weight,
                            surv_mse_loss_weight=self.surv_mse_loss_weight,
                            kl_loss_weight=self.kl_loss_weight,
                            c_index_lb_weight=self.c_index_lb_weight,
                            surved_lr=self.surved_lr,
                            is_tuning=self.is_tuning,
                            surv_regression_error_func=self.surv_regression_error_func,
                            num_samples=self.num_samples,
                            pretrained_file_path=self.pretrained_file_path
                            )
        if self.verbose:
            self.surv_model.summary()
        self.surv_model.fit(dataset=self.dataset, val_id=self.val_id, test_id=self.test_id,
                       folder_name=self.folder_name, model_file_name=self.model_file_name,
                       patience=self.patience, max_epochs=self.max_epochs, verbose=self.verbose,
                       surved_monitor=self.surved_monitor, surved_mode=self.surved_mode,
                       activate_tensorboard=self.activate_tensorboard, batch_size=self.batch_size)

    def run(self):
        self.surv_model.load_predict_print_all_result(show=False, folder_name=self.folder_name, model_file_name=self.model_file_name, log_file_name=self.log_file_name)
        return self.surv_model


class Experiment:
    def __init__(self,exp_name,
                 events_weight=1, censored_weight=1, surv_mse_loss_weight=1, kl_loss_weight=0.05, c_index_lb_weight=0,
                 latent_size=2,
                 drop_percentage=0, events_only=False, action='drop',
                 drop_feature = None,
                 surved_lr=0.00001,
                 surv_regression_error_func='abs',
                 patience=10000, max_epochs=1000000, verbose=True,
                 surved_monitor='val_cindex', surved_mode='max',
                 num_samples=100,
                 activate_tensorboard=False, batch_size=256, activation='sigmoid',
                 ):
        self.drop_percentage = drop_percentage
        self.events_only = events_only
        self.action = action
        self.drop_feature = drop_feature
        self.activation = activation
        self.dataset = self._get_dataset()
        self.structure = self._get_structure(latent_size=latent_size)
        self.latent_size = latent_size
        self.exp_name = exp_name
        self.events_weight = events_weight
        self.censored_weight = censored_weight
        self.surv_mse_loss_weight = surv_mse_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.c_index_lb_weight = c_index_lb_weight
        self.surved_lr = surved_lr
        self.surv_regression_error_func = surv_regression_error_func
        self.patience = patience
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.surved_monitor = surved_monitor
        self.surved_mode = surved_mode
        self.num_samples = num_samples
        self.activate_tensorboard = activate_tensorboard
        self.batch_size = batch_size

        # self._make_base_dir()

    def fit(self, val_id, test_id, fold_id, is_tuning=True, pretrained_file_path=None):
        base_dir = self._get_base_dir(is_tuning=is_tuning)
        self.dataset = self._get_dataset(train_splits_seed=val_id)
        fld = Fold(
            base_dir=base_dir, fold_id=fold_id, dataset=self.dataset, structure=self.structure,
            events_weight=self.events_weight, censored_weight=self.censored_weight,
            surv_mse_loss_weight=self.surv_mse_loss_weight, kl_loss_weight=self.kl_loss_weight,
            c_index_lb_weight=self.c_index_lb_weight,
            surved_lr=self.surved_lr, is_tuning=is_tuning,
            surv_regression_error_func=self.surv_regression_error_func,
            patience=self.patience, max_epochs=self.max_epochs, verbose=self.verbose,
            surved_monitor=self.surved_monitor, surved_mode=self.surved_mode,
            activate_tensorboard=self.activate_tensorboard, batch_size=self.batch_size, activation=self.activation,
            val_id=val_id, test_id=test_id,
            num_samples=self.num_samples,
            pretrained_file_path=pretrained_file_path
        )
        fld.fit()
        return fld

    def run_fold(self, val_id, test_id, fold_id, is_tuning=True, pretrained_file_path=None):
        fld = self.fit(val_id=val_id, test_id=test_id, fold_id=fold_id, is_tuning=is_tuning, pretrained_file_path=pretrained_file_path)
        fld.run()
        return fld.surv_model

    def run_cv(self, test_val_ids_lst=None, is_tuning=False, pretrained_files_list=None):
        base_dir = self._get_base_dir(is_tuning=is_tuning)
        if test_val_ids_lst is None:
            test_val_ids_lst = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
        train_cindexs = []
        val_cindexs = []
        test_cindexs = []
        train_cindexs_events_only = []
        val_cindexs_events_only = []
        test_cindexs_events_only = []

        y_test_preds = []
        for i, (test_id, val_id) in enumerate(test_val_ids_lst):
            if pretrained_files_list is None:
                surved = self.run_fold(val_id=val_id, test_id=test_id, fold_id=i, is_tuning=is_tuning)
            else:
                surved = self.run_fold(val_id=val_id, test_id=test_id, fold_id=i, is_tuning=is_tuning, pretrained_file_path=pretrained_files_list[i])

            y_test_pred = surved.y_test_pred
            y_test_preds.append(y_test_pred)

            train_cindexs.append(surved.cindex_train)
            val_cindexs.append(surved.cindex_val)
            test_cindexs.append(surved.cindex_test)

            train_cindexs_events_only.append(surved.cindex_train_events_only)
            val_cindexs_events_only.append(surved.cindex_val_events_only)
            test_cindexs_events_only.append(surved.cindex_test_events_only)

        log_file = open(base_dir + self.exp_name + '.log', "w")
        self._write_cv_results(train_cindexs, log_file, 'Train')
        self._write_cv_results(val_cindexs, log_file, 'Val')
        self._write_cv_results(test_cindexs, log_file, 'Test')

        self._write_cv_results(train_cindexs_events_only, log_file, 'Train Events Only')
        self._write_cv_results(val_cindexs_events_only, log_file, 'Val Events Only')
        self._write_cv_results(test_cindexs_events_only, log_file, 'Test Events Only')

        del self.dataset
        del surved
        gc.collect()
        return np.mean(test_cindexs), np.mean(val_cindexs), np.mean(train_cindexs), test_cindexs, val_cindexs, train_cindexs, np.array(y_test_preds)

    def run_final_test(self, val_ids_lst=None, is_tuning=False, pretrained_files_list=None):
        base_dir = self._get_base_dir(is_tuning=is_tuning)
        if val_ids_lst is None:
            val_ids_lst = [0, 1, 2, 3, 4]
        train_cindexs = []
        val_cindexs = []
        test_cindexs = []
        train_cindexs_events_only = []
        val_cindexs_events_only = []
        test_cindexs_events_only = []

        y_test_preds = []
        for i, (val_id) in enumerate(val_ids_lst):
            if pretrained_files_list is None:
                surved = self.run_fold(val_id=val_id, test_id=None, fold_id=i, is_tuning=is_tuning)
            else:
                surved = self.run_fold(val_id=val_id, test_id=None, fold_id=i, is_tuning=is_tuning, pretrained_file_path=pretrained_files_list[i])

            y_test_pred = surved.y_test_pred
            y_test_preds.append(y_test_pred)

            train_cindexs.append(surved.cindex_train)
            val_cindexs.append(surved.cindex_val)
            test_cindexs.append(surved.cindex_test)

            train_cindexs_events_only.append(surved.cindex_train_events_only)
            val_cindexs_events_only.append(surved.cindex_val_events_only)
            test_cindexs_events_only.append(surved.cindex_test_events_only)

        log_file = open(base_dir + self.exp_name + '.log', "w")
        self._write_cv_results(train_cindexs, log_file, 'Train')
        self._write_cv_results(val_cindexs, log_file, 'Val')
        self._write_cv_results(test_cindexs, log_file, 'Test')

        self._write_cv_results(train_cindexs_events_only, log_file, 'Train Events Only')
        self._write_cv_results(val_cindexs_events_only, log_file, 'Val Events Only')
        self._write_cv_results(test_cindexs_events_only, log_file, 'Test Events Only')

        del self.dataset
        del surved
        gc.collect()
        return np.mean(test_cindexs), np.mean(val_cindexs), np.mean(train_cindexs), test_cindexs, val_cindexs, train_cindexs, np.array(y_test_preds)

    def run_kl_loss_weight_range(self, values, sub_name='', val_id=0, test_id=1, is_tuning=True):
        base_dir = self._get_base_dir(is_tuning=is_tuning)
        base_dir = base_dir + '/' + 'kl_loss_weight' + ('_' + sub_name + '/' if sub_name != '' else '/')
        for i, value in enumerate(values):
            exp = Fold(
                base_dir=base_dir, fold_id=i, dataset=self.dataset, structure=self.structure,
                events_weight=self.events_weight, censored_weight=self.censored_weight,
                surv_mse_loss_weight=self.surv_mse_loss_weight, kl_loss_weight=value,
                c_index_lb_weight=self.c_index_lb_weight,
                surved_lr=self.surved_lr, is_tuning=is_tuning,
                surv_regression_error_func=self.surv_regression_error_func,
                patience=self.patience, max_epochs=self.max_epochs, verbose=self.verbose,
                surved_monitor=self.surved_monitor, surved_mode=self.surved_mode,
                activate_tensorboard=self.activate_tensorboard, batch_size=self.batch_size,
                activation=self.activation,
                val_id=val_id, test_id=test_id,
                num_samples=self.num_samples,
                pretrained_file_path=None
            )
            exp.run()

    def _get_dataset(self, train_splits_seed=20):
        pass

    def _get_structure(self, latent_size):
        encoder_structure = {
            'Layers': [[32, self.activation, 0.5],
                       [32, self.activation, 0]
                       ],
            'Output_Size': latent_size
        }
        decoder_y_structure = {
            'Layers': [
            ],
            'Output_Size': 1
        }

        structure = {'encoder': encoder_structure, 'decoder_y': decoder_y_structure}

        return structure

    def _get_base_dir(self, is_tuning):
        base_dir = 'Results_' + self.dataset.get_dataset_name() + '/'
        if is_tuning:
            base_dir += 'Tunning/' +self.exp_name + '/'
        else:
            base_dir += 'Experiments/' + self.exp_name + '/'
        return base_dir

    @staticmethod
    def _write_cv_results(lstcv, f, label):
        lstcv_str = [str(i) for i in lstcv]
        strcv = ','.join(lstcv_str)
        f.write(label + ':\n')
        f.write(strcv)
        f.write('\n')
        mu = np.mean(lstcv) * 100
        sm = 2.78 * (np.std(lstcv) * 100) / np.sqrt(len(lstcv))
        f.write('C_Index {:.2f} ({:.2f}, {:.2f})\n'.format(mu, mu - sm, mu + sm))
        f.write('\n')


class FlchainExperiment(Experiment):
    def _get_structure(self, latent_size):
        encoder_structure = {
            'Layers': [[32, self.activation, 0.5],
                       [32, self.activation, 0]
                       ],
            'Output_Size': latent_size
        }
        decoder_y_structure = {
            'Layers': [
            ],
            'Output_Size': 1
        }

        structure = {'encoder': encoder_structure, 'decoder_y': decoder_y_structure}

        return structure

    def _get_dataset(self, train_splits_seed=20):
        return Flchain('Data/flchain.csv', test_fract=0.3, number_of_splits=10, train_splits_seed=train_splits_seed)


class MetabricExperiment(Experiment):
    def _get_structure(self, latent_size):
        encoder_structure = {
            'Layers': [[32, self.activation, 0.5],
                       [32, self.activation, 0]
                       ],
            'Output_Size': latent_size
        }
        decoder_y_structure = {
            'Layers': [
            ],
            'Output_Size': 1
        }

        structure = {'encoder': encoder_structure, 'decoder_y': decoder_y_structure}

        return structure

    def _get_dataset(self, train_splits_seed=20):
        return Metabric("Data/METABRIC.csv", test_fract=0.3, number_of_splits=10, train_splits_seed=train_splits_seed)


class NwtcoExperiment(Experiment):
    def _get_structure(self, latent_size):
        encoder_structure = {
            'Layers': [[32, self.activation, 0.5],
                       [32, self.activation, 0]
                       ],
            'Output_Size': latent_size
        }
        decoder_y_structure = {
            'Layers': [
            ],
            'Output_Size': 1
        }

        structure = {'encoder': encoder_structure, 'decoder_y': decoder_y_structure}

        return structure

    def _get_dataset(self, train_splits_seed=20):
        return Nwtco('Data/nwtco.csv', test_fract=0.3, number_of_splits=10, train_splits_seed=train_splits_seed)


class SupportExperiment(Experiment):
    def _get_structure(self, latent_size):
        encoder_structure = {
            'Layers': [[32, self.activation, 0.5],
                       [32, self.activation, 0]
                       ],
            'Output_Size': latent_size  # 4
        }
        decoder_y_structure = {
            'Layers': [
            ],
            'Output_Size': 1
        }

        structure = {'encoder': encoder_structure, 'decoder_y': decoder_y_structure}

        return structure

    def _get_dataset(self, train_splits_seed=20):
        return Support('Data/support2.csv', p=self.drop_percentage, events_only=self.events_only, action=self.action,
                         drop_feature=self.drop_feature, test_fract=0.3, number_of_splits=10, train_splits_seed=train_splits_seed)
