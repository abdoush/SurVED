import os
import pprint
import sys
import logging

# DATE repository should be downloaded from: "https://github.com/paidamoyo/adversarial_time_to_event" and placed in the same folder.
from flags_parameters import set_params
from model.date import DATE

from my_data.dataset import Flchain, Metabric, Nwtco, Support

if __name__ == '__main__':
    GPUID = "-1"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
    logging.basicConfig(filename='training.log', filemode='a', level=logging.DEBUG)



    model = DATE

    flags = set_params()
    flags.DEFINE_string("path_large_data", '', "path to save folder")
    FLAGS = flags.FLAGS
    pp = pprint.PrettyPrinter()
    pp.pprint(FLAGS.__flags)

    args = sys.argv[1:]
    print("args:{}".format(args))
    if args:
        vm = float(args[0])
    else:
        vm = 1.0
    print("gpu_memory_fraction:{}".format(vm))

    for val_id in range(100):
        flchain = {"path": '', "preprocess": Flchain('my_data/flchain.csv', test_fract=0.3, normalize_target=False, number_of_splits=10, train_splits_seed=val_id), "epochs": 600}
        # support = {"path": '', "preprocess": Support('my_data/support2.csv', test_fract=0.3, normalize_target=False,number_of_splits=10, train_splits_seed=val_id), "epochs": 400}
        # metabric = {"path": '', "preprocess": Metabric('my_data/metabric.csv', test_fract=0.3, normalize_target=False, number_of_splits=10, train_splits_seed=val_id), "epochs": 600}
        # nwtco = {"path": '', "preprocess": Nwtco('my_data/nwtco.csv', test_fract=0.3, normalize_target=False, number_of_splits=10, train_splits_seed=val_id), "epochs": 600}
        dataset = flchain

        data_set = dataset['preprocess'].get_train_val_test_from_splits_for_date_final_eval(val_id=0)
        logging.debug('temp train {}, valid {}, test {}'.format(data_set['train']['x'].shape,
                                                                data_set['valid']['x'].shape,
                                                                data_set['test']['x'].shape))

        print('Round %d===============================================================' % val_id)
        logging.debug('Round %d===============================================================' % val_id)
        logging.debug('Val id: {}, Test id: {}'.format(val_id, 'test set'))


        logging.debug('train {}, valid {}, test {}'.format(data_set['train']['x'].shape, data_set['valid']['x'].shape,
                                                           data_set['test']['x'].shape))

        train_data, valid_data, test_data, end_t, covariates, one_hot_indices, imputation_values \
            = data_set['train'], \
              data_set['valid'], \
              data_set['test'], \
              data_set['end_t'], \
              data_set['covariates'], \
              data_set[
                  'one_hot_indices'], \
              data_set[
                  'imputation_values']

        print("imputation_values:{}, one_hot_indices:{}".format(imputation_values, one_hot_indices))
        print("end_t:{}".format(end_t))
        train = {'x': train_data['x'], 'e': train_data['e'], 't': train_data['t']}
        valid = {'x': valid_data['x'], 'e': valid_data['e'], 't': valid_data['t']}
        test = {'x': test_data['x'], 'e': test_data['e'], 't': test_data['t']}

        perfomance_record = []

        date = model(batch_size=FLAGS.batch_size,
                     learning_rate=FLAGS.learning_rate,
                     beta1=FLAGS.beta1,
                     beta2=FLAGS.beta2,
                     require_improvement=FLAGS.require_improvement,
                     num_iterations=FLAGS.num_iterations, seed=FLAGS.seed,
                     l2_reg=FLAGS.l2_reg,
                     hidden_dim=FLAGS.hidden_dim,
                     train_data=train, test_data=test, valid_data=valid,
                     input_dim=train['x'].shape[1],
                     num_examples=train['x'].shape[0], keep_prob=FLAGS.keep_prob,
                     latent_dim=FLAGS.latent_dim, end_t=end_t,
                     path_large_data=FLAGS.path_large_data,
                     covariates=covariates,
                     categorical_indices=one_hot_indices,
                     disc_updates=FLAGS.disc_updates,
                     sample_size=FLAGS.sample_size, imputation_values=imputation_values,
                     max_epochs=dataset['epochs'], gen_updates=FLAGS.gen_updates)

        with date.session:
            date.train_test()
