from Utils.helper import configure_logger
import math
import os
import sys
import pandas as pd
import random
from Data.dataset import Flchain, Support, Metabric, Nwtco
from MyUtils.metrics import c_index_decomposition
import numpy as np
#import seaborn as sns
import tensorflow as tff
import tensorflow.compat.v1 as tf
from tensorflow import keras
import tf_slim as slim
import logging

# VSI Regpository should be downloaded from https://github.com/ZidiXiu/VSI and placed in the same folder

from utils.preprocessing import formatted_data1, normalize_batch, event_t_bin_prob,risk_t_bin_prob,\
batch_t_categorize, next_batch, one_hot_encoder, one_hot_indices, flatten_nested
from utils.metrics import calculate_quantiles, random_multinomial, MVNloglikeli_np, random_uniform_p
# simulation settings

#tf.disable_v2_behavior()
tf.disable_eager_execution()


import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


def vsi_fit(ds, val_id=1, test_id=0, nc=100, epochs=500, patience=100, final_test=False):
    name = f'cVAE_q_{ds.get_dataset_name()}'
    ### on GPU server
    # directory of output model
    output_dir = f'{currentdir}/results/{ds.get_dataset_name()}/saved_models' + '/'
    #     log_file = output_dir+name+'.log'
    #     logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)

    logger = configure_logger(ds=ds, logdir=currentdir)
    # directory of output test results
    out_dir = f'{currentdir}/results/{ds.get_dataset_name()}' + '/'

    training = True
    # test_id, val_id = (0, 1)

    tf.reset_default_graph()
    seed_num = 22
    np.random.seed(seed_num)
    random.seed(seed_num)
    tff.random.set_seed(seed_num)

    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.keras.backend.set_session(sess)
    # my_init = keras.initializers.glorot_uniform(seed=seed_num)

    #     tf.random.set_random_seed(20)
    #     tf.set_random_seed(20)
    #     tff.random.set_seed(20)

    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    train = formatted_data1(x=x_train, t=y_train, e=e_train)
    test = formatted_data1(x=x_test, t=y_test, e=e_test)
    valid = formatted_data1(x=x_val, t=y_val, e=e_val)

    ## Model hyperparameters
    m = 256
    w_e = 1
    w_ne = 1
    num_sample = 100
    lr = 1e-3


    ncov = train['x'].shape[1]

    # split training time based on bins
    if nc == 'full':
        nbin = int(np.max(y_train))
    else:
        nbin = nc
    tt = np.percentile(train['t'][train['e'] == 1], np.linspace(0., 100., nbin, endpoint=True))
    # based on whether we have censoring after the largest observed t
    loss_of_info = np.mean(train['t'] > np.max(train['t'][train['e'] == 1]))

    # need to convert t to different size of bins
    if loss_of_info > 0.0001:
        print('Yes')
        nbin = nbin + 1
        # add the largest observed censoring time inside
        tt = np.append(tt, np.max(train['t']))
        event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)

    else:
        # get empirical event rate for re-weighting censoring objects
        event_tt_bin, event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)

    # define encoder and decoder
    # slim = tf.contrib.slim
    sample_size = 50
    # start with 3 layers each

## original structure:
    def encoder0(x, is_training):
        """learned prior: Network p(z|x)"""
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            #                         normalizer_fn=slim.batch_norm,
                            #                         normalizer_params={'is_training': is_training},
                            weights_initializer=slim.xavier_initializer()):
            mu_logvar = slim.fully_connected(x, 64, scope='fc1')
            mu_logvar = slim.fully_connected(mu_logvar, 64, scope='fc2')
            mu_logvar = slim.fully_connected(mu_logvar, 64, activation_fn=None, scope='fc3')

        return mu_logvar

    def encoder(x, t_, is_training):
        """Network q(z|x,t_)"""
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            #                         normalizer_fn=slim.batch_norm,
                            #                         normalizer_params={'is_training': is_training},
                            weights_initializer=slim.xavier_initializer()):
            inputs = tf.concat([t_, x], axis=1)
            mu_logvar = slim.fully_connected(inputs, 64, scope='fc1')
            mu_logvar = slim.fully_connected(mu_logvar, 64, scope='fc2')
            mu_logvar = slim.fully_connected(mu_logvar, 64, activation_fn=None, scope='fc3')

        return mu_logvar

    def encoder_z(mu_logvar, epsilon=None):

        # Interpret z as concatenation of mean and log variance
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)

        # Standard deviation must be positive
        stddev = tf.sqrt(tf.exp(logvar))

        if epsilon is None:
            # Draw a z from the distribution
            epsilon = tf.random_normal(tf.shape(stddev))

        z = mu + tf.multiply(stddev, epsilon)

        return z

    def decoder(z, is_training):
        """Network p(t|z)"""
        # Decoding arm
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            #                         normalizer_fn=slim.batch_norm,
                            #                         normalizer_params={'is_training': is_training},
                            weights_initializer=slim.xavier_initializer()):
            t_logits = slim.fully_connected(z, 64, scope='fc1')
            t_logits = slim.fully_connected(t_logits, 64, scope='fc2')
            t_logits = slim.fully_connected(t_logits, 64, scope='fc3')

            # returns multinomial distribution
            t_logits = slim.fully_connected(t_logits, nbin, activation_fn=None, scope='fc4')
            # t_logits = tf.nn.softmax(t_logits)

        return (t_logits)


    def VAE_losses(t_logits, t_truncate, mu_logvar0, mu_logvar1, tiny=1e-8):
        # NEW ONE! with different strategy of calculating loss for censoring, adding \sum p_b, not \sum w_b*p_b
        """Define loss functions (reconstruction, KL divergence) and optimizer"""
        # Reconstruction loss
        t_dist = tf.nn.softmax(t_logits)
        reconstruction = -tf.log(tf.reduce_sum(t_dist * t_truncate, axis=1))

        # KL divergence
        mu0, logvar0 = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
        mu1, logvar1 = tf.split(mu_logvar1, num_or_size_splits=2, axis=1)

        kl_d = 0.5 * tf.reduce_sum(tf.exp(logvar1 - logvar0) \
                                   + tf.divide(tf.square(mu0 - mu1), tf.exp(logvar0) + tiny) \
                                   + logvar0 - logvar1 - 1.0, \
                                   1)

        # Total loss for event
        loss = tf.reduce_mean(reconstruction + kl_d)

        return reconstruction, kl_d, loss

    def pt_x(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training):
        # here t_ is known!
        # for calculation purposes, censoring subject t_ need to be a truncated form like [0,0,0,1,1,1]
        # which could calculete sum of all bins after censoring time
        mu, logvar = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
        # sample z_l
        # q_{\beta}(z_l|t_i,x_i)
        epsilon = tf.random_normal(tf.shape(logvar))
        z1_sample = encoder_z(mu_logvar, epsilon)
        # only have one dimension here
        t_logits_l = decoder(z1_sample, is_training)
        t_dist_l = tf.nn.softmax(t_logits_l)
        p_t_z = tf.reduce_sum(t_truncate * t_dist_l, 1)
        pq_z = tf.exp(MVNloglikeli(z1_sample, mu_logvar0, noise=1e-8) \
                      - MVNloglikeli(z1_sample, mu_logvar, noise=1e-8))
        pt_x_l = p_t_z * pq_z
        pt_x_sum = pt_x_l

        for k in range(num_sample - 1):
            # q_{\beta}(z_l|t_i,x_i)
            epsilon = tf.random_normal(tf.shape(logvar))
            z1_sample = encoder_z(mu_logvar, epsilon)
            #         # p_{\alpha}(t_i|z_l)
            #         epsilon = tf.random_normal(tf.shape(logvar))
            #         z0_sample = encoder_z(mu_logvar0, epsilon)
            #         # p_{\alpha}(z_l|x)
            #         epsilon = tf.random_normal(tf.shape(logvar))
            #         # only have one dimension here
            t_logits_l = decoder(z1_sample, is_training)
            t_dist_l = tf.nn.softmax(t_logits_l)
            p_t_z = tf.reduce_sum(t_truncate * t_dist_l, 1)
            pq_z = tf.exp(MVNloglikeli(z1_sample, mu_logvar0, noise=1e-8) \
                          - MVNloglikeli(z1_sample, mu_logvar, noise=1e-8))
            pt_x_l = p_t_z * pq_z

            # sum up
            pt_x_sum = pt_x_sum + pt_x_l

        pt_x_avg = pt_x_sum / num_sample
        return (pt_x_avg)

    def loglikeli_cVAE(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training):
        pt_x_avg = pt_x(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training)
        return (tf.log(pt_x_avg))

    # MVN log-likelihood
    def MVNloglikeli(z, mu_logvar, noise=1e-8):
        # Interpret z as concatenation of mean and log variance
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)

        # note that Sigma is a diagonal matrix and we only have the diagonal information here
        varmatrix = tf.exp(logvar)

        # calculate log-likelihood
        #     likeli = -0.5*(tf.log(tf.linalg.det(varmatrix)+noise)\
        #                    +tf.matmul(tf.matmul((z-mu), tf.linalg.inv(varmatrix))\
        #                              ,tf.transpose(z-mu))\
        #                    +nbin*np.log(2*np.pi)
        #                   )
        # for diagonal matrix:
        loglikeli = -0.5 * (tf.log(varmatrix) + (z - mu) ** 2 / varmatrix + np.log(2 * np.pi))
        # returns a log-likelihood for each z
        return tf.reduce_sum(loglikeli, axis=1)

    def t_dist_avg(mu_logvar0, t_logits_init, num_sample, is_training):
        mu, logvar = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
        t_dist_new_sum = tf.nn.softmax(t_logits_init)
        for k in range(num_sample - 1):
            # graph resample basic implementation
            epsilon = tf.random_normal(tf.shape(logvar))
            t_logits_new_k = decoder(encoder_z(mu_logvar0, epsilon), is_training)
            t_dist_new_k = tf.nn.softmax(t_logits_new_k)
            t_dist_new_sum = t_dist_new_sum + t_dist_new_k
        t_dist_new_avg = tf.math.divide(t_dist_new_sum, num_sample)
        return (t_dist_new_avg)

    def zero_outputs():
        # just to return 3 outputs to match previous function for events instead
        return 0.0, 0.0, 0.0

    ####Main Structure
    # training indicator
    is_training = tf.placeholder(tf.bool, [], name="is_training");

    # Define input placeholder
    t_ = tf.placeholder(tf.float32, [None, nbin], name='t_')
    # Define input placeholder only for calculating likelihood or survival function purpose
    t_truncate = tf.placeholder(tf.float32, [None, nbin], name='t_truncate')

    # each patient will only have 1 indicator of censoring or event
    event = tf.placeholder(tf.float32, [None], name='event')
    x = tf.placeholder(tf.float32, [None, ncov], name='x')

    # separate the input as event and censoring
    # we still keep observations in original order
    e_idx = tf.where(tf.equal(event, 1.))
    e_idx = tf.reshape(e_idx, [tf.shape(e_idx)[0]])
    ne_idx = tf.where(tf.equal(event, 0.))
    ne_idx = tf.reshape(ne_idx, [tf.shape(ne_idx)[0]])

    e_is_empty = tf.equal(tf.size(e_idx), 0)
    ne_is_empty = tf.equal(tf.size(ne_idx), 0)

    # Define VAE graph
    with tf.variable_scope('encoder0'):
        # update parameters encoder0 for all observations
        mu_logvar0 = encoder0(x, is_training)
        z0 = encoder_z(mu_logvar0)

    # update encoder q for both censoring and events
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        # with events, true t is t_;
        # for censoring, true time is t_r
        mu_logvar1 = encoder(x, t_, is_training)
        z1 = encoder_z(mu_logvar1)

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        # update for all samples
        t_logits_1 = decoder(z1, is_training)
        # update for all samples
        t_logits_0 = decoder(z0, is_training)

        # predict posterior distribution based on multiple z
        t_dist_new = tf.nn.softmax(t_logits_0)
        # Calculating average distribution
        t_dist_new_avg = t_dist_avg(mu_logvar0, t_dist_new, num_sample, is_training)

        # calculate likelihood based on randomly sample multiple z1
        event_loglikeli = loglikeli_cVAE(tf.gather(t_truncate, e_idx), tf.gather(mu_logvar0, e_idx),
                                         tf.gather(mu_logvar1, e_idx), num_sample, is_training)
        censor_loglikeli = loglikeli_cVAE(tf.gather(t_truncate, ne_idx), tf.gather(mu_logvar0, ne_idx),
                                          tf.gather(mu_logvar1, ne_idx), num_sample, is_training)

        total_loglikeli = loglikeli_cVAE(t_truncate, mu_logvar0, mu_logvar1, num_sample, is_training)
    # Optimization
    with tf.variable_scope('training') as scope:
        # calculate the losses separately, just for debugging purposes
        # calculate losses for events
        e_recon, e_kl_d, eloss = tf.cond(e_is_empty, lambda: zero_outputs(), \
                                         lambda: VAE_losses(tf.gather(t_logits_1, e_idx), tf.gather(t_truncate, e_idx), \
                                                            tf.gather(mu_logvar0, e_idx), tf.gather(mu_logvar1, e_idx)))

        # calculate losses for censor
        ne_recon, ne_kl_d, closs = tf.cond(ne_is_empty, lambda: zero_outputs(), \
                                           lambda: VAE_losses(tf.gather(t_logits_1, ne_idx),
                                                              tf.gather(t_truncate, ne_idx), \
                                                              tf.gather(mu_logvar0, ne_idx),
                                                              tf.gather(mu_logvar1, ne_idx)))

        loss = w_e * eloss + w_ne * closs
        #         print(f'w_e: {w_e}, w_ec: {w_ne}')
        #         print(f'Loss: {loss}')
        #         tf.print(loss, output_stream=sys.stdout)
        #         print(f'Loses: {ne_recon}, {ne_kl_d}, {closs}')
        #         tf.print(ne_recon, output_stream=sys.stdout)
        #         tf.print(ne_kl_d, output_stream=sys.stdout)
        #         tf.print(closs, output_stream=sys.stdout)

        # compute together
        rec_all, kl_d_all, loss_all = VAE_losses(t_logits_1, t_truncate, mu_logvar0, mu_logvar1)
        #    train_step_unlabeled = tf.train.AdamOptimizer().minimize(loss)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(loss_all, params)
        # gradients = tf.Print(gradients,[gradients], message ='gradients',summarize=2000)
        grads = zip(gradients, params)

        # optimizer = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.9, beta2=0.999)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999)
        train_step = optimizer.apply_gradients(grads)

    def wAvg_t(sess, new_x, post_prob, tt, num_sample, return_wi=False):
        # calculate weighted average
        for j in range(num_sample):
            t_hat_l = np.array([random_uniform_p(tt, post_prob[subj], 1) for subj in range(post_prob.shape[0])])
            t_hat_binned = batch_t_categorize(t_hat_l, np.ones(t_hat_l.shape), tt, event_tt_prob=1.0)
            mu_logvar0l = sess.run(mu_logvar0, feed_dict={x: new_x, is_training: False})
            mu_logvar1l = sess.run(mu_logvar1, feed_dict={x: new_x, t_: t_hat_binned, is_training: False})
            # sample z1l
            mu1l, logvar1l = np.split(mu_logvar1l, 2, 1)
            epsilon_l = np.random.normal(size=logvar1l.shape)
            # Standard deviation must be positive
            stddevl = np.sqrt(np.exp(logvar1l))
            z1l = mu1l + np.multiply(stddevl, epsilon_l)
            ## calculate weight
            wil = np.divide(np.exp(MVNloglikeli_np(z1l, mu_logvar0l, noise=1e-8)), \
                            np.exp(MVNloglikeli_np(z1l, mu_logvar1l, noise=1e-8)))
            if j == 0:
                t_hat_all = np.array(t_hat_l).reshape(post_prob.shape[0], 1)
                wl_all = wil.reshape(post_prob.shape[0], 1)
            else:
                t_hat_all = np.concatenate([t_hat_all, np.array(t_hat_l).reshape(post_prob.shape[0], 1)], axis=1)
                wl_all = np.concatenate([wl_all, wil.reshape(post_prob.shape[0], 1)], axis=1)

        t_hat_i = np.sum(np.multiply(t_hat_all, wl_all), axis=1) / np.sum(wl_all, axis=1)
        if return_wi == False:
            return t_hat_i
        else:
            return (t_hat_i, np.mean(wl_all, axis=1), np.std(wl_all, axis=1))

    def saveResults(dataset, session_dir, session_name, out_dir, tt, event_tt_prob):
        sess = tf.Session()
        session_path = session_dir + session_name + ".ckpt"
        saver.restore(sess, session_path)
        # run over all samples in test
        batch_x, batch_t, batch_e = dataset['x'], dataset['t'], dataset['e']
        batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

        batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
        norm_batch_x = batch_x.copy()
        # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
        test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training: False})
        test_loglikeli = sess.run(total_loglikeli,
                                  feed_dict={t_truncate: batch_t_cat_likeli, t_: batch_t_cat, x: norm_batch_x,
                                             event: batch_e, is_training: False})
        # this provide likelihood
        #     test_pt_x_avg = sess.run(total_pt_x_avg, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:batch_x, event:batch_e, is_training:False})
        test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample,
                                                               return_wi=True)

        test_pred_medt = [calculate_quantiles(post_prob, tt, 0.5) for post_prob in test_pred_prob]
        test_pred_medt = np.concatenate(test_pred_medt, axis=0)
        test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])
        np.save(out_dir + '/{}_test_pred_prob'.format(session_name), test_pred_prob)
        np.save(out_dir + '/{}_test_loglikeli'.format(session_name), test_loglikeli)
        np.save(out_dir + '/{}_test_pred_avgt'.format(session_name), test_pred_avgt)
        np.save(out_dir + '/{}_test_pred_medt'.format(session_name), test_pred_medt)
        np.save(out_dir + '/{}_test_pred_randomt'.format(session_name), test_pred_randomt)
        np.save(out_dir + '/{}_tt'.format(session_name), tt)

    def saveResults_norun(session_name, out_dir, tt, test_pred_prob, test_loglikeli, test_pred_avgt, test_pred_medt,
                          test_pred_randomt):
        np.save(out_dir + '/{}_test_pred_prob'.format(session_name), test_pred_prob)
        np.save(out_dir + '/{}_test_loglikeli'.format(session_name), test_loglikeli)
        np.save(out_dir + '/{}_test_pred_avgt'.format(session_name), test_pred_avgt)
        np.save(out_dir + '/{}_test_pred_medt'.format(session_name), test_pred_medt)
        np.save(out_dir + '/{}_test_pred_randomt'.format(session_name), test_pred_randomt)
        np.save(out_dir + '/{}_tt'.format(session_name), tt)

    ##########################
    #### Training ############
    ##########################
    if training == True:
        valid_recon_loss = []
        valid_epoch_recon_loss = []
        valid_epoch_loss = []
        valid_epoch_event_recon_loss = []
        valid_epoch_censor_recon_loss = []

        best_likelihood = -np.inf
        best_i = 0
        best_epoch = 0
        num_epoch = epochs  # 200 abdo
        num_sample = 100  # for sampling
        num_batch = int(train['x'].shape[0] / m)
        require_impr = patience
        saver = tf.train.Saver()
        # event_tt_prob = event_t_bin_prob_unif(tt)

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        # tf.keras.backend.set_session(sess)
        # my_init = keras.initializers.glorot_uniform(seed=seed_num)

        with tf.Session() as sess:

            tf.keras.backend.set_session(sess)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Train VAE model
            for i in range(num_epoch * num_batch):
                # Get a training minibatch
                batch_x, batch_t, batch_e = next_batch(train, m=m)
                batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
                # normalize input
                norm_batch_x = batch_x.copy()
                # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
                # Binarize the data
                batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)
                # Train on minibatch
                sess.run(train_step,
                         feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x, event: batch_e,
                                    is_training: True})
                # sess.run(train_step_SGD, feed_dict={t_:batch_t_cat, x:batch_x, event:batch_e, is_training:True})

                if i % num_batch == 0:
                    batch_x, batch_t, batch_e = next_batch(valid, m=valid['x'].shape[0])
                    batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)
                    batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
                    norm_batch_x = batch_x.copy()
                    # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)

                    epoch_loglikeli = np.mean(
                        sess.run(total_loglikeli, feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli, \
                                                             x: norm_batch_x, event: batch_e, is_training: False}))
                    epoch_loss = sess.run(loss_all,
                                          feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x,
                                                     event: batch_e, is_training: False})

                    valid_epoch_recon_loss.append(epoch_loglikeli)
                    valid_epoch_loss.append(epoch_loss)
                    epoch_recon_closs = np.mean(sess.run(ne_recon,
                                                         feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli,
                                                                    x: norm_batch_x, event: batch_e,
                                                                    is_training: False}))
                    valid_epoch_censor_recon_loss.append(epoch_recon_closs)
                    epoch_recon_eloss = np.mean(sess.run(e_recon,
                                                         feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli,
                                                                    x: norm_batch_x, event: batch_e,
                                                                    is_training: False}))
                    valid_epoch_event_recon_loss.append(epoch_recon_eloss)
                    if (best_likelihood <= epoch_loglikeli):
                        best_likelihood = epoch_loglikeli
                        best_i = i
                        # save the learned model
                        save_path = saver.save(sess, output_dir + name + ".ckpt")

                    op_print = ('Epoch ' + str(i / num_batch) + ': Loss ' + str(epoch_loss) \
                                + ' log-likelihood: ' + str(epoch_loglikeli) \
                                + ' event rec loss: ' + str(epoch_recon_eloss) \
                                + ' censor rec loss: ' + str(epoch_recon_closs))
                    logging.debug(op_print)

                # early stopping
                if (i - best_i) > require_impr:
                    print("Model stops improving for a while")
                    break
        ##### return results on testing dataset #####
        # run over all samples in test
        saveResults(test, session_dir=output_dir, session_name=name, out_dir=out_dir, tt=tt,
                    event_tt_prob=event_tt_prob)



    #### only for testing #####
    else:
        sess = tf.Session()
        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(sess, output_dir + name + ".ckpt")
        # run over all samples in test

        # run over all samples in test
        batch_x, batch_t, batch_e = test['x'], test['t'], test['e']
        batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

        batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)

        norm_batch_x = batch_x.copy()
        # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
        test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training: False})
        test_loglikeli = sess.run(total_loglikeli,
                                  feed_dict={t_truncate: batch_t_cat_likeli, t_: batch_t_cat, x: norm_batch_x,
                                             event: batch_e, is_training: False})
        test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample,
                                                               return_wi=True)

        test_pred_medt = [calculate_quantiles(post_prob, tt, 0.5) for post_prob in test_pred_prob]
        test_pred_medt = np.concatenate(test_pred_medt, axis=0)
        test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])

        saveResults_norun(session_name=name, out_dir=out_dir, tt=tt, test_pred_prob=test_pred_prob,
                          test_loglikeli=test_loglikeli, test_pred_avgt=test_pred_avgt, test_pred_medt=test_pred_medt,
                          test_pred_randomt=test_pred_randomt)

    #     if val:
    #         dataset = valid
    #     else:
    #         dataset = test
    dataset = test

    session_dir = output_dir
    session_name = name
    # out_dir=out_dir
    # tt=tt
    # event_tt_prob=event_tt_prob

    sess = tf.Session()
    session_path = session_dir + session_name + ".ckpt"
    saver.restore(sess, session_path)
    # run over all samples in test
    batch_x, batch_t, batch_e = dataset['x'], dataset['t'], dataset['e']
    batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

    batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
    norm_batch_x = batch_x.copy()
    # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
    test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training: False})
    test_loglikeli = sess.run(total_loglikeli,
                              feed_dict={t_truncate: batch_t_cat_likeli, t_: batch_t_cat, x: norm_batch_x,
                                         event: batch_e, is_training: False})
    # this provide likelihood
    #     test_pt_x_avg = sess.run(total_pt_x_avg, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:batch_x, event:batch_e, is_training:False})
    test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample,
                                                           return_wi=True)

    test_pred_medt = [calculate_quantiles(post_prob, tt, 0.5) for post_prob in test_pred_prob]
    test_pred_medt = np.concatenate(test_pred_medt, axis=0)
    test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])

    t_true = batch_t
    e_true = batch_e
    t_pred = test_pred_avgt

    c_ee, c_ec, alpha, alpha_deviation, c = c_index_decomposition(t_true, t_pred, e_true)

    print(f"c_ee:{c_ee}, c_ec:{c_ec}, alpha:{alpha}, alpha_deviation:{alpha_deviation}, c:{c}")

    return c, t_pred


def vsi_cv(nc, ds, epochs=500, patience=100, final_test=False):
    cis = []
    y_preds = []
    for val_id, test_id in [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]:
        ci, y_pred = vsi_fit(nc=nc, ds=ds, test_id=test_id, val_id=val_id, epochs=epochs, patience=patience,
                             final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)

    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def vsi_cv_nfolds(nc, ds_class, ds_file_name, epochs=500, patience=100, final_test=False, n_folds=100):
    cis = []
    y_preds = []
    for val_id in range(n_folds):
        ds = ds_class(f'{parentdir}/Data/{ds_file_name}.csv', normalize_target=False, test_fract=0.3, number_of_splits=10, train_splits_seed=val_id)
        ci, y_pred = vsi_fit(nc=nc, ds=ds, test_id=0, val_id=0, epochs=epochs, patience=patience, final_test=final_test)
        cis.append(ci)
        y_preds.append(y_pred)
        print(f'{val_id}: {ci}')
    if final_test:
        return cis, np.array(y_preds)

    return np.mean(cis)


def random_search(ds, epochs=500, patience=100, logdir=None):
    logger = configure_logger(ds, logdir)
    best_c_index = 0
    best_nc = 0
    for nc in [100, 200, 400, 1000]:
        logger.info(f'Testing:, {nc}')
        c_index = vsi_cv(nc=nc, ds=ds, epochs=epochs, patience=patience, final_test=False)
        logger.info(c_index)
        if (c_index > best_c_index):
            best_c_index = c_index
            logger.info(f'New best c-index: {c_index}')
            logger.info('=================================================================')
            best_nc = nc
    return best_nc


def vsi_fit_change_censoring(ds, val_id=1, test_id=0, nc=100, epochs=500, patience=100, final_test=False):
    name = f'cVAE_q_{ds.get_dataset_name()}'
    ### on GPU server
    # directory of output model
    output_dir = f'{currentdir}/results/{ds.get_dataset_name()}/saved_models' + '/'
    #     log_file = output_dir+name+'.log'
    #     logging.basicConfig(filename=log_file, filemode='w', level=logging.DEBUG)

    logger = configure_logger(ds=ds, logdir=currentdir)
    # directory of output test results
    out_dir = f'{currentdir}/results/{ds.get_dataset_name()}' + '/'

    training = True
    # test_id, val_id = (0, 1)

    tf.reset_default_graph()
    seed_num = 22
    # np.random.seed(seed_num)
    # random.seed(seed_num)
    tff.random.set_seed(seed_num)
    # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    # sess = tf.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
    # tf.keras.backend.set_session(sess)
    # my_init = keras.initializers.glorot_uniform(seed=seed_num)

    #     tf.random.set_random_seed(20)
    #     tf.set_random_seed(20)
    #     tff.random.set_seed(20)

    if final_test:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_final_eval(val_id=val_id)
    else:
        (x_train, ye_train, y_train, e_train,
         x_val, ye_val, y_val, e_val,
         x_test, ye_test, y_test, e_test) = ds.get_train_val_test_from_splits(test_id=test_id, val_id=val_id)

    train = formatted_data1(x=x_train, t=y_train, e=e_train)
    test = formatted_data1(x=x_test, t=y_test, e=e_test)
    valid = formatted_data1(x=x_val, t=y_val, e=e_val)

    ## Model hyperparameters
    m = 256
    w_e = 1
    w_ne = 1
    num_sample = 100
    lr = 1e-3


    ncov = train['x'].shape[1]

    # split training time based on bins
    if nc == 'full':
        nbin = int(np.max(y_train))
    else:
        nbin = nc
    tt = np.percentile(train['t'][train['e'] == 1], np.linspace(0., 100., nbin, endpoint=True))
    # based on whether we have censoring after the largest observed t
    loss_of_info = np.mean(train['t'] > np.max(train['t'][train['e'] == 1]))

    # need to convert t to different size of bins
    if loss_of_info > 0.0001:
        print('Yes')
        nbin = nbin + 1
        # add the largest observed censoring time inside
        tt = np.append(tt, np.max(train['t']))
        event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)

    else:
        # get empirical event rate for re-weighting censoring objects
        event_tt_bin, event_tt_prob = risk_t_bin_prob(train['t'], train['e'], tt)

    # define encoder and decoder
    # slim = tf.contrib.slim
    sample_size = 50
    # start with 3 layers each

## original structure:
    def encoder0(x, is_training):
        """learned prior: Network p(z|x)"""
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            #                         normalizer_fn=slim.batch_norm,
                            #                         normalizer_params={'is_training': is_training},
                            weights_initializer=slim.xavier_initializer()):
            mu_logvar = slim.fully_connected(x, 64, scope='fc1')
            mu_logvar = slim.fully_connected(mu_logvar, 64, scope='fc2')
            mu_logvar = slim.fully_connected(mu_logvar, 64, activation_fn=None, scope='fc3')

        return mu_logvar

    def encoder(x, t_, is_training):
        """Network q(z|x,t_)"""
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            #                         normalizer_fn=slim.batch_norm,
                            #                         normalizer_params={'is_training': is_training},
                            weights_initializer=slim.xavier_initializer()):
            inputs = tf.concat([t_, x], axis=1)
            mu_logvar = slim.fully_connected(inputs, 64, scope='fc1')
            mu_logvar = slim.fully_connected(mu_logvar, 64, scope='fc2')
            mu_logvar = slim.fully_connected(mu_logvar, 64, activation_fn=None, scope='fc3')

        return mu_logvar

    def encoder_z(mu_logvar, epsilon=None):

        # Interpret z as concatenation of mean and log variance
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)

        # Standard deviation must be positive
        stddev = tf.sqrt(tf.exp(logvar))

        if epsilon is None:
            # Draw a z from the distribution
            epsilon = tf.random_normal(tf.shape(stddev))

        z = mu + tf.multiply(stddev, epsilon)

        return z

    def decoder(z, is_training):
        """Network p(t|z)"""
        # Decoding arm
        with slim.arg_scope([slim.fully_connected],
                            activation_fn=tf.nn.leaky_relu,
                            #                         normalizer_fn=slim.batch_norm,
                            #                         normalizer_params={'is_training': is_training},
                            weights_initializer=slim.xavier_initializer()):
            t_logits = slim.fully_connected(z, 64, scope='fc1')
            t_logits = slim.fully_connected(t_logits, 64, scope='fc2')
            t_logits = slim.fully_connected(t_logits, 64, scope='fc3')

            # returns multinomial distribution
            t_logits = slim.fully_connected(t_logits, nbin, activation_fn=None, scope='fc4')
            # t_logits = tf.nn.softmax(t_logits)

        return (t_logits)


    def VAE_losses(t_logits, t_truncate, mu_logvar0, mu_logvar1, tiny=1e-8):
        # NEW ONE! with different strategy of calculating loss for censoring, adding \sum p_b, not \sum w_b*p_b
        """Define loss functions (reconstruction, KL divergence) and optimizer"""
        # Reconstruction loss
        t_dist = tf.nn.softmax(t_logits)
        reconstruction = -tf.log(tf.reduce_sum(t_dist * t_truncate, axis=1))

        # KL divergence
        mu0, logvar0 = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
        mu1, logvar1 = tf.split(mu_logvar1, num_or_size_splits=2, axis=1)

        kl_d = 0.5 * tf.reduce_sum(tf.exp(logvar1 - logvar0) \
                                   + tf.divide(tf.square(mu0 - mu1), tf.exp(logvar0) + tiny) \
                                   + logvar0 - logvar1 - 1.0, \
                                   1)

        # Total loss for event
        loss = tf.reduce_mean(reconstruction + kl_d)

        return reconstruction, kl_d, loss

    def pt_x(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training):
        # here t_ is known!
        # for calculation purposes, censoring subject t_ need to be a truncated form like [0,0,0,1,1,1]
        # which could calculete sum of all bins after censoring time
        mu, logvar = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
        # sample z_l
        # q_{\beta}(z_l|t_i,x_i)
        epsilon = tf.random_normal(tf.shape(logvar))
        z1_sample = encoder_z(mu_logvar, epsilon)
        # only have one dimension here
        t_logits_l = decoder(z1_sample, is_training)
        t_dist_l = tf.nn.softmax(t_logits_l)
        p_t_z = tf.reduce_sum(t_truncate * t_dist_l, 1)
        pq_z = tf.exp(MVNloglikeli(z1_sample, mu_logvar0, noise=1e-8) \
                      - MVNloglikeli(z1_sample, mu_logvar, noise=1e-8))
        pt_x_l = p_t_z * pq_z
        pt_x_sum = pt_x_l

        for k in range(num_sample - 1):
            # q_{\beta}(z_l|t_i,x_i)
            epsilon = tf.random_normal(tf.shape(logvar))
            z1_sample = encoder_z(mu_logvar, epsilon)
            #         # p_{\alpha}(t_i|z_l)
            #         epsilon = tf.random_normal(tf.shape(logvar))
            #         z0_sample = encoder_z(mu_logvar0, epsilon)
            #         # p_{\alpha}(z_l|x)
            #         epsilon = tf.random_normal(tf.shape(logvar))
            #         # only have one dimension here
            t_logits_l = decoder(z1_sample, is_training)
            t_dist_l = tf.nn.softmax(t_logits_l)
            p_t_z = tf.reduce_sum(t_truncate * t_dist_l, 1)
            pq_z = tf.exp(MVNloglikeli(z1_sample, mu_logvar0, noise=1e-8) \
                          - MVNloglikeli(z1_sample, mu_logvar, noise=1e-8))
            pt_x_l = p_t_z * pq_z

            # sum up
            pt_x_sum = pt_x_sum + pt_x_l

        pt_x_avg = pt_x_sum / num_sample
        return (pt_x_avg)

    def loglikeli_cVAE(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training):
        pt_x_avg = pt_x(t_truncate, mu_logvar0, mu_logvar, num_sample, is_training)
        return (tf.log(pt_x_avg))

    # MVN log-likelihood
    def MVNloglikeli(z, mu_logvar, noise=1e-8):
        # Interpret z as concatenation of mean and log variance
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)

        # note that Sigma is a diagonal matrix and we only have the diagonal information here
        varmatrix = tf.exp(logvar)

        # calculate log-likelihood
        #     likeli = -0.5*(tf.log(tf.linalg.det(varmatrix)+noise)\
        #                    +tf.matmul(tf.matmul((z-mu), tf.linalg.inv(varmatrix))\
        #                              ,tf.transpose(z-mu))\
        #                    +nbin*np.log(2*np.pi)
        #                   )
        # for diagonal matrix:
        loglikeli = -0.5 * (tf.log(varmatrix) + (z - mu) ** 2 / varmatrix + np.log(2 * np.pi))
        # returns a log-likelihood for each z
        return tf.reduce_sum(loglikeli, axis=1)

    def t_dist_avg(mu_logvar0, t_logits_init, num_sample, is_training):
        mu, logvar = tf.split(mu_logvar0, num_or_size_splits=2, axis=1)
        t_dist_new_sum = tf.nn.softmax(t_logits_init)
        for k in range(num_sample - 1):
            # graph resample basic implementation
            epsilon = tf.random_normal(tf.shape(logvar))
            t_logits_new_k = decoder(encoder_z(mu_logvar0, epsilon), is_training)
            t_dist_new_k = tf.nn.softmax(t_logits_new_k)
            t_dist_new_sum = t_dist_new_sum + t_dist_new_k
        t_dist_new_avg = tf.math.divide(t_dist_new_sum, num_sample)
        return (t_dist_new_avg)

    def zero_outputs():
        # just to return 3 outputs to match previous function for events instead
        return 0.0, 0.0, 0.0

    ####Main Structure
    # training indicator
    is_training = tf.placeholder(tf.bool, [], name="is_training");

    # Define input placeholder
    t_ = tf.placeholder(tf.float32, [None, nbin], name='t_')
    # Define input placeholder only for calculating likelihood or survival function purpose
    t_truncate = tf.placeholder(tf.float32, [None, nbin], name='t_truncate')

    # each patient will only have 1 indicator of censoring or event
    event = tf.placeholder(tf.float32, [None], name='event')
    x = tf.placeholder(tf.float32, [None, ncov], name='x')

    # separate the input as event and censoring
    # we still keep observations in original order
    e_idx = tf.where(tf.equal(event, 1.))
    e_idx = tf.reshape(e_idx, [tf.shape(e_idx)[0]])
    ne_idx = tf.where(tf.equal(event, 0.))
    ne_idx = tf.reshape(ne_idx, [tf.shape(ne_idx)[0]])

    e_is_empty = tf.equal(tf.size(e_idx), 0)
    ne_is_empty = tf.equal(tf.size(ne_idx), 0)

    # Define VAE graph
    with tf.variable_scope('encoder0'):
        # update parameters encoder0 for all observations
        mu_logvar0 = encoder0(x, is_training)
        z0 = encoder_z(mu_logvar0)

    # update encoder q for both censoring and events
    with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
        # with events, true t is t_;
        # for censoring, true time is t_r
        mu_logvar1 = encoder(x, t_, is_training)
        z1 = encoder_z(mu_logvar1)

    with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
        # update for all samples
        t_logits_1 = decoder(z1, is_training)
        # update for all samples
        t_logits_0 = decoder(z0, is_training)

        # predict posterior distribution based on multiple z
        t_dist_new = tf.nn.softmax(t_logits_0)
        # Calculating average distribution
        t_dist_new_avg = t_dist_avg(mu_logvar0, t_dist_new, num_sample, is_training)

        # calculate likelihood based on randomly sample multiple z1
        event_loglikeli = loglikeli_cVAE(tf.gather(t_truncate, e_idx), tf.gather(mu_logvar0, e_idx),
                                         tf.gather(mu_logvar1, e_idx), num_sample, is_training)
        censor_loglikeli = loglikeli_cVAE(tf.gather(t_truncate, ne_idx), tf.gather(mu_logvar0, ne_idx),
                                          tf.gather(mu_logvar1, ne_idx), num_sample, is_training)

        total_loglikeli = loglikeli_cVAE(t_truncate, mu_logvar0, mu_logvar1, num_sample, is_training)
    # Optimization
    with tf.variable_scope('training') as scope:
        # calculate the losses separately, just for debugging purposes
        # calculate losses for events
        e_recon, e_kl_d, eloss = tf.cond(e_is_empty, lambda: zero_outputs(), \
                                         lambda: VAE_losses(tf.gather(t_logits_1, e_idx), tf.gather(t_truncate, e_idx), \
                                                            tf.gather(mu_logvar0, e_idx), tf.gather(mu_logvar1, e_idx)))

        # calculate losses for censor
        ne_recon, ne_kl_d, closs = tf.cond(ne_is_empty, lambda: zero_outputs(), \
                                           lambda: VAE_losses(tf.gather(t_logits_1, ne_idx),
                                                              tf.gather(t_truncate, ne_idx), \
                                                              tf.gather(mu_logvar0, ne_idx),
                                                              tf.gather(mu_logvar1, ne_idx)))

        loss = w_e * eloss + w_ne * closs
        #         print(f'w_e: {w_e}, w_ec: {w_ne}')
        #         print(f'Loss: {loss}')
        #         tf.print(loss, output_stream=sys.stdout)
        #         print(f'Loses: {ne_recon}, {ne_kl_d}, {closs}')
        #         tf.print(ne_recon, output_stream=sys.stdout)
        #         tf.print(ne_kl_d, output_stream=sys.stdout)
        #         tf.print(closs, output_stream=sys.stdout)

        # compute together
        rec_all, kl_d_all, loss_all = VAE_losses(t_logits_1, t_truncate, mu_logvar0, mu_logvar1)
        #    train_step_unlabeled = tf.train.AdamOptimizer().minimize(loss)
        params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        gradients = tf.gradients(loss_all, params)
        # gradients = tf.Print(gradients,[gradients], message ='gradients',summarize=2000)
        grads = zip(gradients, params)

        # optimizer = tf.train.AdamOptimizer(learning_rate=5e-4, beta1=0.9, beta2=0.999)
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999)
        train_step = optimizer.apply_gradients(grads)

    def wAvg_t(sess, new_x, post_prob, tt, num_sample, return_wi=False):
        # calculate weighted average
        for j in range(num_sample):
            t_hat_l = np.array([random_uniform_p(tt, post_prob[subj], 1) for subj in range(post_prob.shape[0])])
            t_hat_binned = batch_t_categorize(t_hat_l, np.ones(t_hat_l.shape), tt, event_tt_prob=1.0)
            mu_logvar0l = sess.run(mu_logvar0, feed_dict={x: new_x, is_training: False})
            mu_logvar1l = sess.run(mu_logvar1, feed_dict={x: new_x, t_: t_hat_binned, is_training: False})
            # sample z1l
            mu1l, logvar1l = np.split(mu_logvar1l, 2, 1)
            epsilon_l = np.random.normal(size=logvar1l.shape)
            # Standard deviation must be positive
            stddevl = np.sqrt(np.exp(logvar1l))
            z1l = mu1l + np.multiply(stddevl, epsilon_l)
            ## calculate weight
            wil = np.divide(np.exp(MVNloglikeli_np(z1l, mu_logvar0l, noise=1e-8)), \
                            np.exp(MVNloglikeli_np(z1l, mu_logvar1l, noise=1e-8)))
            if j == 0:
                t_hat_all = np.array(t_hat_l).reshape(post_prob.shape[0], 1)
                wl_all = wil.reshape(post_prob.shape[0], 1)
            else:
                t_hat_all = np.concatenate([t_hat_all, np.array(t_hat_l).reshape(post_prob.shape[0], 1)], axis=1)
                wl_all = np.concatenate([wl_all, wil.reshape(post_prob.shape[0], 1)], axis=1)

        t_hat_i = np.sum(np.multiply(t_hat_all, wl_all), axis=1) / np.sum(wl_all, axis=1)
        if return_wi == False:
            return t_hat_i
        else:
            return (t_hat_i, np.mean(wl_all, axis=1), np.std(wl_all, axis=1))

    def saveResults(dataset, session_dir, session_name, out_dir, tt, event_tt_prob):
        sess = tf.Session()
        session_path = session_dir + session_name + ".ckpt"
        saver.restore(sess, session_path)
        # run over all samples in test
        batch_x, batch_t, batch_e = dataset['x'], dataset['t'], dataset['e']
        batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

        batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
        norm_batch_x = batch_x.copy()
        # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
        test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training: False})
        test_loglikeli = sess.run(total_loglikeli,
                                  feed_dict={t_truncate: batch_t_cat_likeli, t_: batch_t_cat, x: norm_batch_x,
                                             event: batch_e, is_training: False})
        # this provide likelihood
        #     test_pt_x_avg = sess.run(total_pt_x_avg, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:batch_x, event:batch_e, is_training:False})
        test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample,
                                                               return_wi=True)

        test_pred_medt = [calculate_quantiles(post_prob, tt, 0.5) for post_prob in test_pred_prob]
        test_pred_medt = np.concatenate(test_pred_medt, axis=0)
        test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])
        np.save(out_dir + '/{}_test_pred_prob'.format(session_name), test_pred_prob)
        np.save(out_dir + '/{}_test_loglikeli'.format(session_name), test_loglikeli)
        np.save(out_dir + '/{}_test_pred_avgt'.format(session_name), test_pred_avgt)
        np.save(out_dir + '/{}_test_pred_medt'.format(session_name), test_pred_medt)
        np.save(out_dir + '/{}_test_pred_randomt'.format(session_name), test_pred_randomt)
        np.save(out_dir + '/{}_tt'.format(session_name), tt)

    def saveResults_norun(session_name, out_dir, tt, test_pred_prob, test_loglikeli, test_pred_avgt, test_pred_medt,
                          test_pred_randomt):
        np.save(out_dir + '/{}_test_pred_prob'.format(session_name), test_pred_prob)
        np.save(out_dir + '/{}_test_loglikeli'.format(session_name), test_loglikeli)
        np.save(out_dir + '/{}_test_pred_avgt'.format(session_name), test_pred_avgt)
        np.save(out_dir + '/{}_test_pred_medt'.format(session_name), test_pred_medt)
        np.save(out_dir + '/{}_test_pred_randomt'.format(session_name), test_pred_randomt)
        np.save(out_dir + '/{}_tt'.format(session_name), tt)

    ##########################
    #### Training ############
    ##########################
    if training == True:
        valid_recon_loss = []
        valid_epoch_recon_loss = []
        valid_epoch_loss = []
        valid_epoch_event_recon_loss = []
        valid_epoch_censor_recon_loss = []

        best_likelihood = -np.inf
        best_i = 0
        best_epoch = 0
        num_epoch = epochs  # 200 abdo
        num_sample = 100  # for sampling
        num_batch = int(train['x'].shape[0] / m)
        require_impr = patience
        saver = tf.train.Saver()
        # event_tt_prob = event_t_bin_prob_unif(tt)

        session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
        # sess = tf.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
        # tf.keras.backend.set_session(sess)
        # my_init = keras.initializers.glorot_uniform(seed=seed_num)

        with tf.Session() as sess:

            tf.keras.backend.set_session(sess)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # Train VAE model
            for i in range(num_epoch * num_batch):
                # Get a training minibatch
                batch_x, batch_t, batch_e = next_batch(train, m=m)
                batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
                # normalize input
                norm_batch_x = batch_x.copy()
                # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
                # Binarize the data
                batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)
                # Train on minibatch
                sess.run(train_step,
                         feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x, event: batch_e,
                                    is_training: True})
                # sess.run(train_step_SGD, feed_dict={t_:batch_t_cat, x:batch_x, event:batch_e, is_training:True})

                if i % num_batch == 0:
                    batch_x, batch_t, batch_e = next_batch(valid, m=valid['x'].shape[0])
                    batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)
                    batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
                    norm_batch_x = batch_x.copy()
                    # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)

                    epoch_loglikeli = np.mean(
                        sess.run(total_loglikeli, feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli, \
                                                             x: norm_batch_x, event: batch_e, is_training: False}))
                    epoch_loss = sess.run(loss_all,
                                          feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli, x: norm_batch_x,
                                                     event: batch_e, is_training: False})

                    valid_epoch_recon_loss.append(epoch_loglikeli)
                    valid_epoch_loss.append(epoch_loss)
                    epoch_recon_closs = np.mean(sess.run(ne_recon,
                                                         feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli,
                                                                    x: norm_batch_x, event: batch_e,
                                                                    is_training: False}))
                    valid_epoch_censor_recon_loss.append(epoch_recon_closs)
                    epoch_recon_eloss = np.mean(sess.run(e_recon,
                                                         feed_dict={t_: batch_t_cat, t_truncate: batch_t_cat_likeli,
                                                                    x: norm_batch_x, event: batch_e,
                                                                    is_training: False}))
                    valid_epoch_event_recon_loss.append(epoch_recon_eloss)
                    if (best_likelihood <= epoch_loglikeli):
                        best_likelihood = epoch_loglikeli
                        best_i = i
                        # save the learned model
                        save_path = saver.save(sess, output_dir + name + ".ckpt")

                    op_print = ('Epoch ' + str(i / num_batch) + ': Loss ' + str(epoch_loss) \
                                + ' log-likelihood: ' + str(epoch_loglikeli) \
                                + ' event rec loss: ' + str(epoch_recon_eloss) \
                                + ' censor rec loss: ' + str(epoch_recon_closs))
                    logging.debug(op_print)

                # early stopping
                if (i - best_i) > require_impr:
                    print("Model stops improving for a while")
                    break
        ##### return results on testing dataset #####
        # run over all samples in test
        saveResults(test, session_dir=output_dir, session_name=name, out_dir=out_dir, tt=tt,
                    event_tt_prob=event_tt_prob)



    #### only for testing #####
    else:
        sess = tf.Session()
        # Restore variables from disk.
        saver = tf.train.Saver()
        saver.restore(sess, output_dir + name + ".ckpt")
        # run over all samples in test

        # run over all samples in test
        batch_x, batch_t, batch_e = test['x'], test['t'], test['e']
        batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

        batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)

        norm_batch_x = batch_x.copy()
        # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
        test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training: False})
        test_loglikeli = sess.run(total_loglikeli,
                                  feed_dict={t_truncate: batch_t_cat_likeli, t_: batch_t_cat, x: norm_batch_x,
                                             event: batch_e, is_training: False})
        test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample,
                                                               return_wi=True)

        test_pred_medt = [calculate_quantiles(post_prob, tt, 0.5) for post_prob in test_pred_prob]
        test_pred_medt = np.concatenate(test_pred_medt, axis=0)
        test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])

        saveResults_norun(session_name=name, out_dir=out_dir, tt=tt, test_pred_prob=test_pred_prob,
                          test_loglikeli=test_loglikeli, test_pred_avgt=test_pred_avgt, test_pred_medt=test_pred_medt,
                          test_pred_randomt=test_pred_randomt)

    #     if val:
    #         dataset = valid
    #     else:
    #         dataset = test
    dataset = test

    session_dir = output_dir
    session_name = name
    # out_dir=out_dir
    # tt=tt
    # event_tt_prob=event_tt_prob

    sess = tf.Session()
    session_path = session_dir + session_name + ".ckpt"
    saver.restore(sess, session_path)
    # run over all samples in test
    batch_x, batch_t, batch_e = dataset['x'], dataset['t'], dataset['e']
    batch_t_cat = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob)

    batch_t_cat_likeli = batch_t_categorize(batch_t, batch_e, tt, event_tt_prob, likelihood=True)
    norm_batch_x = batch_x.copy()
    # abd norm_batch_x[:,cts_covariates] = normalize_batch(batch_x[:,cts_covariates],norm_mean,norm_std)
    test_pred_prob = sess.run(t_dist_new_avg, feed_dict={x: norm_batch_x, is_training: False})
    test_loglikeli = sess.run(total_loglikeli,
                              feed_dict={t_truncate: batch_t_cat_likeli, t_: batch_t_cat, x: norm_batch_x,
                                         event: batch_e, is_training: False})
    # this provide likelihood
    #     test_pt_x_avg = sess.run(total_pt_x_avg, feed_dict={t_truncate:batch_t_cat_likeli, t_:batch_t_cat, x:batch_x, event:batch_e, is_training:False})
    test_pred_avgt, test_avgt_mean, test_avgt_std = wAvg_t(sess, norm_batch_x, test_pred_prob, tt, num_sample,
                                                           return_wi=True)

    test_pred_medt = [calculate_quantiles(post_prob, tt, 0.5) for post_prob in test_pred_prob]
    test_pred_medt = np.concatenate(test_pred_medt, axis=0)
    test_pred_randomt = np.array([random_uniform_p(tt, post_prob, 1) for post_prob in test_pred_prob])

    t_true = batch_t
    e_true = batch_e
    t_pred = test_pred_avgt

    c_ee, c_ec, alpha, alpha_deviation, c = c_index_decomposition(t_true, t_pred, e_true)

    print(f"c_ee:{c_ee}, c_ec:{c_ec}, alpha:{alpha}, alpha_deviation:{alpha_deviation}, c:{c}")

    return c, t_pred, y_test, e_test


def vsi_change_size_only(nc, epochs=500, patience=100, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.60, 0.51, 0.36, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='drop', events_only=False)
        ci, y_pred, y_test, e_test = vsi_fit_change_censoring(ds=ds_support, val_id=0, test_id=1, nc=nc, epochs=epochs, patience=patience, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/vsi_final_results_change_size_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)


def vsi_change_censoring_only(nc, epochs=500, patience=100, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='censor', events_only=True)
        ci, y_pred, y_test, e_test = vsi_fit_change_censoring(ds=ds_support, val_id=0, test_id=1, nc=nc, epochs=epochs, patience=patience, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/vsi_final_results_change_censoring_only_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)


def vsi_change_censoring_and_size(nc, epochs=500, patience=100, final_test=True):
    cis = []
    y_preds = []
    # for changing the size only [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0]
    # for changing the events (drop or censor)[0.20, 0.35, 0.50, 'full']:
    # pe = 0.50

    for pe in [0.20, 0.35, 0.50, 'full']:
        ds_support = Support(f'{parentdir}/Data/support2.csv', normalize_target=False, test_fract=0.3, p=pe, action='drop', events_only=True)
        ci, y_pred, y_test, e_test = vsi_fit_change_censoring(ds=ds_support, val_id=0, test_id=1, nc=nc, epochs=epochs, patience=patience, final_test=final_test)
        cis.append(ci)
        df = pd.DataFrame()
        df['y_pred'] = y_pred
        df['y_test'] = y_test
        df['e_test'] = e_test
        df.to_csv(f'{currentdir}/vsi_final_results_change_censoring_and_size_{pe}.csv', index=False)
    print(cis)
    return np.mean(cis)

