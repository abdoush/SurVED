from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, BatchNormalization
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import math
from lifelines.utils import concordance_index
from Utils.metrics import c_index_decomposition
import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
import datetime
import pandas as pd
from sksurv.nonparametric import SurvivalFunctionEstimator
import random
from tensorflow import keras


class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim), seed=1)
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class BaseModel:
    def __init__(self, model_input, model_structure_dict, name, stocastic_output=False):
        self.model = self._buid_model(model_input, model_structure_dict, stocastic_output, name)

    @staticmethod
    def _buid_model(model_input, model_structure_dict, stocastic_output, name):
        seed_num = 1
        os.environ['PYTHONHASHSEED'] = '0'
        np.random.seed(seed_num)
        #random.seed(seed_num)
        tf.random.set_seed(seed_num)
        session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        tf.compat.v1.keras.backend.set_session(sess)
        my_init = keras.initializers.glorot_uniform(seed=seed_num)
        output_size = model_structure_dict['Output_Size']
        x = model_input
        for i, (layer_size, layer_activation, layer_dropout) in enumerate(model_structure_dict['Layers']):
            x = Dense(layer_size, name=name+'_'+str(i), kernel_initializer=my_init)(x)
            #x = BatchNormalization()(x)
            if layer_activation != '':
                x = Activation(layer_activation)(x)
            if layer_dropout > 0:
                x = Dropout(layer_dropout)(x)
        if stocastic_output:
            z_mean = Dense(output_size, name=name+'_z_mean', kernel_initializer=my_init)(x)
            z_log_var = Dense(output_size, name=name+'_z_z_log_var', kernel_initializer=my_init)(x)
            z = Sampling()((z_mean, z_log_var))
            model_output = [z_mean, z_log_var, z]
        else:
            model_output = Dense(output_size, name=name+'_out', kernel_initializer=my_init)(x) # , activation='sigmoid' , activation='relu'

        model = Model(model_input, model_output, name=name)
        return model


class SurVED:
    def __init__(self, input_shape, encoder_structure_dict, decoder_y_structure_dict,
                 activation,
                 events_weight, censored_weight, surv_mse_loss_weight, kl_loss_weight, c_index_lb_weight,
                 surved_lr,
                 is_tuning=True,
                 surv_regression_error_func='abs',
                 num_samples=200,
                 pretrained_file_path=None
                 ):
        self.enc_input = Input(input_shape)
        self.dec_input = Input(encoder_structure_dict['Output_Size'])
        self.input_shape = input_shape
        self.activation = activation
        self.events_weight = events_weight
        self.censored_weight = censored_weight
        self.surv_mse_loss_weight = surv_mse_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.c_index_lb_weight = c_index_lb_weight
        self.surv_regression_error_func = surv_regression_error_func
        self.is_fitted = False
        self.surved_history = None
        self.is_tuning = is_tuning
        self.surved_lr = surved_lr
        self.num_samples = num_samples
        self.encoder = BaseModel(self.enc_input, encoder_structure_dict, stocastic_output=True, name='Encoder').model
        self.decoder_y = BaseModel(self.dec_input, decoder_y_structure_dict, stocastic_output=False, name='Decoder').model
        self.pretrained_file_path = pretrained_file_path
        self.surved_model = self._build_compile(self.enc_input)

    def _build_compile(self, model_input):
        z_mean, z_log_var, z = self.encoder(model_input)
        surved_y_output = self.decoder_y(z)
        surved = Model(model_input, surved_y_output, name='SurVED')

        kl_loss_orig = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        kl_loss = kl_loss_orig * self.kl_loss_weight
        surved.add_loss(K.mean(kl_loss))
        surved.add_metric(kl_loss_orig, name='kl_loss', aggregation='mean')
        opt = Adam(lr=self.surved_lr)
        surved.compile(loss=self._get_loss(), optimizer=opt, metrics=[self.cindex, self.surv_mse_loss])
        return surved

    def fit(self, dataset, val_id, test_id, folder_name, model_file_name, patience=20000, max_epochs=1000000, verbose=0,
            surved_monitor='val_cindex', surved_mode='max',
            activate_tensorboard=False, batch_size=256,
            ):
        self.patience = patience
        self.val_id = val_id
        self.test_id = test_id
        self.surved_monitor = surved_monitor
        self.surved_mode = surved_mode
        self.max_epochs = max_epochs
        self.verbose = verbose
        self.activate_tensorboard = activate_tensorboard
        self.batch_size = batch_size
        self.dataset = dataset

        if test_id is None:
            (self.x_train, self.ye_train, self.y_train, self.e_train,
             self.x_val, self.ye_val, self.y_val, self.e_val,
             self.x_test, self.ye_test, self.y_test, self.e_test) = dataset.get_train_val_test_final_eval(val_id=0)
        else:
            (self.x_train, self.ye_train, self.y_train, self.e_train,
             self.x_val, self.ye_val, self.y_val, self.e_val,
             self.x_test, self.ye_test, self.y_test, self.e_test) = dataset.get_train_val_test_from_splits(val_id, test_id)
        print('x_train', self.x_train.shape)
        print('x_val', self.x_val.shape)
        print('x_test', self.x_test.shape)

        self.timeline = self._get_timeline(self.y_train)

        self.surved_callbacks = self._get_callbacks(folder_name=folder_name,
                                                    model_file_name=model_file_name,
                                                    monitor=surved_monitor,
                                                    patience=patience,
                                                    mode=surved_mode, activate_tensorboard=activate_tensorboard)

        if self.pretrained_file_path is None:
            hist = self.surved_model.fit(self.x_train, self.ye_train,
                                         validation_data=(self.x_val, self.ye_val),
                                         callbacks=self.surved_callbacks,
                                         epochs=self.max_epochs,
                                         verbose=self.verbose,
                                         batch_size=self.batch_size) #math.ceil(self.x_train.shape[0] / self.num_batches))
            self.surved_history = hist
        else:
            self.load_model(self.pretrained_file_path)

        self.is_fitted = True

    def _get_callbacks(self, folder_name, model_file_name, monitor, patience, mode, activate_tensorboard):
        file_name = 'SurVED_best.h5'
        log_dir = os.path.join(
            "logs",
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )
        if activate_tensorboard:
            callbacks = [EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode=mode, baseline=0),
                         ModelCheckpoint(filepath=folder_name + model_file_name, monitor=monitor, save_best_only=True, mode=mode),
                         TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq=100000)]
        else:
            callbacks = [EarlyStopping(monitor=monitor, patience=patience, restore_best_weights=True, mode=mode, baseline=0),
                         ModelCheckpoint(filepath=folder_name + model_file_name, monitor=monitor, save_best_only=True, mode=mode)]
        return callbacks

    @staticmethod
    def cindex_lowerbound(y_true, y_pred):
        y = y_true[:, 0]
        e = y_true[:, 1]
        ydiff = y[tf.newaxis, :] - y[:, tf.newaxis]
        yij = K.cast(K.greater(ydiff, 0), K.floatx()) + K.cast(K.equal(ydiff, 0), K.floatx()) * K.cast(
            e[:, tf.newaxis] != e[tf.newaxis, :], K.floatx())  # yi > yj
        is_valid_pair = yij * e[:, tf.newaxis]

        ypdiff = tf.transpose(y_pred) - y_pred  # y_pred[tf.newaxis,:] - y_pred[:,tf.newaxis]
        ypij = (1 + K.log(K.sigmoid(ypdiff))) / K.log(tf.constant(2.0))
        cidx_lb = (K.sum(ypij * is_valid_pair)) / K.sum(is_valid_pair)
        return tf.cond(K.sum(e)==0.0, lambda: 0.5, lambda: cidx_lb) #cidx_lb

    @staticmethod
    def cindex(y_true, y_pred):
        y = y_true[:, 0]
        e = y_true[:, 1]
        ydiff = y[tf.newaxis, :] - y[:, tf.newaxis]
        yij = K.cast(K.greater(ydiff, 0), K.floatx()) + K.cast(K.equal(ydiff, 0), K.floatx()) * K.cast(
            e[:, tf.newaxis] != e[tf.newaxis, :], K.floatx())  # yi > yj
        is_valid_pair = yij * e[:, tf.newaxis]

        ypdiff = tf.transpose(y_pred) - y_pred
        ypij = K.cast(K.greater(ypdiff, 0), K.floatx()) + 0.5 * K.cast(K.equal(ypdiff, 0), K.floatx())  # yi > yj
        cidx = (K.sum(ypij * is_valid_pair)) / K.sum(is_valid_pair)
        #return cidx
        return tf.cond(K.sum(e)==0.0, lambda: 0.5, lambda: cidx)

    @staticmethod
    def _get_history_keys(history):
        return [x for x in history.keys() if not x.startswith('val_')]

    def _plot_histories(self, history, name, folder_name):
        keys = self._get_history_keys(history)
        for key in keys:
            self._plot_history(history, key, name=name, folder_name=folder_name)

    def _plot_history(self, history, key, name, folder_name):
        fig, ax = plt.subplots(1, 1, figsize=(10, 7))
        ax.plot(history[key], label=key)
        ax.plot(history['val_'+key], label='val_'+key)
        ax.set_title(name + ' ' + key)
        ax.set_xlabel('Epochs')
        ax.set_ylabel(key)
        ax.legend()
        plt.savefig(folder_name+name+'_'+key+'.png')

    def _plot_y_histogram(self, folder_name):
        if not self.is_tuning:
            n_sub_figs = 3
        else:
            n_sub_figs = 2
        fig, ax = plt.subplots(1, n_sub_figs, figsize=(n_sub_figs*7, 5))
        ax[0].hist(self.y_train[self.e_train==0], color='C0', alpha=0.3, label='censored')
        ax[0].hist(self.y_train[self.e_train==1], color='C1', alpha=0.3, label='event')
        ax[0].set_title('Train')
        ax[0].legend()
        ax[1].hist(self.y_val[self.e_val == 0], color='C0', alpha=0.3, label='censored')
        ax[1].hist(self.y_val[self.e_val == 1], color='C1', alpha=0.3, label='event')
        ax[1].set_title('Val')
        ax[1].legend()
        if not self.is_tuning:
            ax[2].hist(self.y_test[self.e_test == 0], color='C0', alpha=0.3, label='censored')
            ax[2].hist(self.y_test[self.e_test == 1], color='C1', alpha=0.3, label='event')
            ax[2].set_title('Test')
            ax[2].legend()
        plt.savefig(folder_name + 'y_hist.png')

    def load_predict_print_all_result(self,folder_name, model_file_name, log_file_name, show=False):
        if not self.is_fitted:
            print('Model is not fitted yet!')
            return

        if self.pretrained_file_path is None:
            self.load_model(folder_name + model_file_name)
        else:
            self.load_model(self.pretrained_file_path)
        self.get_results()
        self._save_results_as_dataframe(folder_name=folder_name)
        #self._save_survival_functions_as_dataframe(folder_name=folder_name)
        if self.verbose:
            self.print_cindexs()
        if self.surved_history is not None:
            self._plot_histories(self.surved_history.history, name='SurVED', folder_name=folder_name)
            with open(folder_name + 'surved_history.dic', 'wb') as file_pi:
                pickle.dump(self.surved_history.history, file_pi)

        #self._plot_y_histogram(folder_name=folder_name)
        #self.plot_train_val_test_2d_latent_space(folder_name=folder_name)
        #self.plot_train_val_test_latent_var_hist(folder_name=folder_name)
        self.save_experiment_settings(folder_name=folder_name, log_file_name=log_file_name)
        if show:
            plt.show()
        else:
            plt.close('all')

    def _save_results_as_dataframe(self, folder_name):
        if self.is_tuning:
            df = pd.DataFrame({'t': self.y_val, 'e': self.e_val, 'p': self.y_val_pred})
        else:
            df = pd.DataFrame({'t': self.y_test, 'e': self.e_test, 'p': self.y_test_pred})
        df.to_csv(folder_name + 'results.csv', index=False)

    def _save_survival_functions_as_dataframe(self, folder_name):
        if self.is_tuning:
            df = pd.DataFrame(self.s_val_pred)
        else:
            df = pd.DataFrame(self.s_test_pred)
        df.to_csv(folder_name + 'results_survival_functions.csv', index=False)

        timeline_df = pd.DataFrame(self.timeline)
        timeline_df.to_csv(folder_name + 'results_timeline.csv', index=False)

    def load_model(self, filepath):
        self.surved_model.load_weights(filepath)

    def get_results(self):
        if not self.is_fitted:
            print('Model is not fitted yet!')
            return

        self.y_train_pred = self.predict(self.x_train, self.num_samples)
        self.y_val_pred = self.predict(self.x_val, self.num_samples)
        self.y_test_pred = self.predict(self.x_test, self.num_samples)

        # self.sf_train = self.get_survival_function(self.x_train, self.num_samples)
        # self.sf_val = self.get_survival_function(self.x_val, self.num_samples)
        # self.sf_test = self.get_survival_function(self.x_test, self.num_samples)
        #
        # self.s_train_pred = self.predict_survival_props(survival_functions=self.sf_train, timeline=self.timeline)
        # self.s_val_pred = self.predict_survival_props(survival_functions=self.sf_val, timeline=self.timeline)
        # self.s_test_pred = self.predict_survival_props(survival_functions=self.sf_test, timeline=self.timeline)

        self.z_mean_train, self.z_log_var_train, self.z_train = self.encoder.predict(self.x_train, self.num_samples)
        self.z_mean_val, self.z_log_var_val, self.z_val = self.encoder.predict(self.x_val, self.num_samples)
        self.z_mean_test, self.z_log_var_test, self.z_test = self.encoder.predict(self.x_test, self.num_samples)

        self.cindex_train = concordance_index(self.y_train, self.y_train_pred, self.e_train.astype(int))
        self.cindex_train_events_only = concordance_index(self.y_train[self.e_train == 1],
                                                          self.y_train_pred[self.e_train == 1],
                                                          self.e_train[self.e_train == 1].astype(int))
        self.cindex_val = concordance_index(self.y_val, self.y_val_pred, self.e_val.astype(int))
        self.cindex_val_events_only = concordance_index(self.y_val[self.e_val == 1],
                                                        self.y_val_pred[self.e_val == 1],
                                                        self.e_val[self.e_val == 1].astype(int))
        self.cindex_test = concordance_index(self.y_test, self.y_test_pred, self.e_test.astype(int))
        self.cindex_test_events_only = concordance_index(self.y_test[self.e_test==1],
                                                         self.y_test_pred[self.e_test==1],
                                                         self.e_test[self.e_test==1].astype(int))

        (c_ee, c_ec, alpha, alpha_deviation, c) = c_index_decomposition(t=self.y_test,
                                                                        y=self.y_test_pred,
                                                                        e=self.e_test)

        self.cid = f'c_ee: {c_ee}, c_ec: {c_ec}, alpha: {alpha}, alpha_deviation: {alpha_deviation}, c: {c}'

    def print_cindexs(self):
        print('Train cindex {:.2f}'.format(self.cindex_train * 100))
        print('Train cindex Event Only {:.2f}'.format(self.cindex_train_events_only * 100))
        print('Val cindex {:.2f}'.format(self.cindex_val * 100))
        print('Val cindex Events Only {:.2f}'.format(self.cindex_val_events_only * 100))
        if not self.is_tuning:
            print('Test cindex {:.2f}'.format(self.cindex_test * 100))
            print('Test cindex Events Only {:.2f}'.format(self.cindex_test_events_only * 100))

    def predict(self, x, num_samples=100):
        scores_y = []
        for _ in range(num_samples):
            score_y = self.surved_model.predict(x)
            scores_y.append(score_y[:, 0])

        return np.median(np.array(scores_y), axis=0) # np.array(scores_y).mean(axis=0) # np.median(np.array(scores_y), axis=0) #

    def get_survival_function(self, x, num_samples=100):
        scores_y = []
        for _ in range(num_samples):
            score_y = self.surved_model.predict(x)
            scores_y.append(score_y[:, 0])
        res = np.array(scores_y)

        survival_functions = []
        for i in range(res.shape[1]):
            times = res[:, i]
            et = np.array([(True, t) for t in times], dtype=np.dtype('bool,float'))
            # print(et)
            s = SurvivalFunctionEstimator()
            s.fit(et)
            survival_functions.append(s)
        return survival_functions

    @staticmethod
    def predict_survival_props(survival_functions, timeline):
        pp = []
        for s in survival_functions:
            p = s.predict_proba(timeline)
            pp.append(p)
        return np.array(pp)  # output shape (num instances, num timesteps), (1, num timesteps)

    def predict_survival_function(self, x, timeline):
        ss = self.get_survival_function(x)
        return self.predict_survival_props(ss, timeline)

    @staticmethod
    def _get_timeline(y):
        return np.array(sorted(np.unique(y)))

    def surv_mse_loss(self, y_true, y_pred):
        e = y_true[:, 1]
        y_diff = (y_true[:, 0] - y_pred[:, 0])
        err_func = getattr(K, self.surv_regression_error_func)
        err = self.events_weight * e * err_func(y_diff) + self.censored_weight * (1 - e) * err_func(K.relu(y_diff))
        return K.mean(err)

    def _get_loss(self):
        if self.surv_mse_loss_weight == 0:
            def _loss(y_true, y_pred):
                return self.surv_mse_loss(y_true, y_pred)*self.surv_mse_loss_weight
        else:
            def _loss(y_true, y_pred):
                return self.surv_mse_loss(y_true, y_pred)*self.surv_mse_loss_weight \
                       - self.cindex_lowerbound(y_true, y_pred)*self.c_index_lb_weight
        return _loss

    def plot_train_val_test_2d_latent_space(self, folder_name):
        if not self.is_tuning:
            n_sub_figs = 3
        else:
            n_sub_figs = 2
        fig, ax = plt.subplots(1, n_sub_figs, figsize=(n_sub_figs * 5, 5))
        self.plot_2d_latent_space(self.z_mean_train, self.e_train, 'z_mean Train', ax[0])
        self.plot_2d_latent_space(self.z_mean_val, self.e_val, 'z_mean Val', ax[1])
        if not self.is_tuning:
            self.plot_2d_latent_space(self.z_mean_test, self.e_test, 'z_mean Test', ax[2])
        plt.savefig(folder_name + 'train_val_test_latent_space.png')

    @staticmethod
    def plot_2d_latent_space(z_mean, e, title, axis):
        axis.scatter(z_mean[e == 0, 0], z_mean[e == 0, 1], alpha=0.2)
        axis.scatter(z_mean[e == 1, 0], z_mean[e == 1, 1], alpha=0.2)
        axis.axhline(y=0, ls=':', c='k')
        axis.axvline(x=0, ls=':', c='k')
        axis.set_xlabel('z0')
        axis.set_ylabel('z1')
        axis.set_title(title)

    def plot_train_val_test_latent_var_hist(self, folder_name):
        self._plot_latent_var_hist(self.z_mean_train, np.exp(self.z_log_var_train), self.e_train, 'train_latent_var', folder_name=folder_name)
        self._plot_latent_var_hist(self.z_mean_val, np.exp(self.z_log_var_val), self.e_val, 'val_latent_var', folder_name=folder_name)
        if not self.is_tuning:
            self._plot_latent_var_hist(self.z_mean_test, np.exp(self.z_log_var_test), self.e_test, 'test_latent_var', folder_name=folder_name)

    def _plot_latent_var_hist(self, z_mean, z_var, e, title, folder_name):
        latent_dim = z_mean.shape[1]
        fig, ax = plt.subplots(2, latent_dim, figsize=(latent_dim * 3, 2 * 3))
        for i in range(latent_dim):
            self._plot_hist(z_mean[:, i], e, 'z_mean %d' % i, ax[0, i])
            self._plot_hist(z_var[:, i], e, 'z_var %d' % i, ax[1, i])
        fig.suptitle(title)
        plt.savefig(folder_name + title + '_hist.png')

    def _plot_hist(self, z, e, title, axis):
        axis.hist(z[e == 0], alpha=0.2)
        axis.hist(z[e == 1], alpha=0.2)
        axis.set_title(title)

    def save_experiment_settings(self, folder_name, log_file_name):
        log_file = open(folder_name + log_file_name, "w")

        log_file.write(self.dataset.print_dataset_summery())
        log_file.write('\n')

        log_file.write('Weights:\n')
        log_file.write('events_weight: {}\n'.format(self.events_weight))
        log_file.write('censored_weight: {}\n'.format(self.censored_weight))
        log_file.write('surv_mse_loss_weight: {}\n'.format(self.surv_mse_loss_weight))
        log_file.write('kl_loss_weight: {}\n'.format(self.kl_loss_weight))
        log_file.write('c_index_lb_weight: {}\n'.format(self.c_index_lb_weight))
        log_file.write('\n')

        log_file.write('activation: {}\n'.format(self.activation))
        log_file.write('surv_regression_error_func: {}\n'.format(self.surv_regression_error_func))
        log_file.write('patience: {}\n'.format(self.patience))
        log_file.write('val_id: {}\n'.format(self.val_id))
        log_file.write('test_id: {}\n'.format(self.test_id))
        log_file.write('monitor: {}\n'.format(self.surved_monitor))
        log_file.write('lr: {}\n'.format(self.surved_lr))
        log_file.write('\n')

        log_file.write('Results\n')
        log_file.write('Cindex Train: {:.2f}\n'.format((self.cindex_train * 100)))
        log_file.write('Cindex Val: {:.2f}\n'.format((self.cindex_val * 100)))
        if not self.is_tuning:
            log_file.write('Cindex Test: {:.2f}\n'.format((self.cindex_test * 100)))
        log_file.write('\n')

        log_file.write('Results Events Only\n')
        log_file.write('Cindex Train Events Only: {:.2f}\n'.format((self.cindex_train_events_only * 100)))
        log_file.write('Cindex Val Events Only: {:.2f}\n'.format((self.cindex_val_events_only * 100)))
        if not self.is_tuning:
            log_file.write('Cindex Test Events Only: {:.2f}\n'.format((self.cindex_test_events_only * 100)))
        log_file.write('\n')
        log_file.write(self.cid)
        log_file.write('\n')
        self.encoder.summary(print_fn=lambda x: log_file.write(x + '\n'))
        log_file.write('\n')
        self.decoder_y.summary(print_fn=lambda x: log_file.write(x + '\n'))
        log_file.write('\n')
        self.surved_model.summary(print_fn=lambda x: log_file.write(x + '\n'))
        log_file.write('\n')

        log_file.close()

    def summary(self):
        self.encoder.summary()
        self.decoder_y.summary()
        self.surved_model.summary()
