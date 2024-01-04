import os
import pprint
import sys
import logging

# DATE repository should be downloaded from: "https://github.com/paidamoyo/adversarial_time_to_event" and placed in the same folder.
from flags_parameters import set_params
from model.date import DATE

from my_data.dataset import Support

if __name__ == '__main__':
    GPUID = "0"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPUID)
    logging.basicConfig(filename='training.log', filemode='a', level=logging.DEBUG)


    # for changing the size only pe ranges in [0.601317957166392, 0.5093904448105436, 0.36210873146622735, 0] and events_only=False
    # for changing the events (drop or censor) pe ranges in [0.20, 0.35, 0.50, 'full'] and events_only=True
    pe = 0.50
    events_only = True
    print(pe)

    support = {"path": '', "preprocess": Support('my_data/support2.csv', p=pe, action='drop', events_only=events_only, test_fract=0.3, normalize_target=False), "epochs": 400}

    dataset = support

    data_set = dataset['preprocess'].get_train_val_test_from_splits_for_date_final_eval(val_id=0)

    model = DATE

    flags = set_params()
    flags.DEFINE_string("path_large_data", dataset['path'], "path to save folder")
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

    i = 0
    val_id = 0
    data_set = dataset['preprocess'].get_train_val_test_from_splits_for_date_final_eval(val_id=val_id)
    logging.debug('temp train {}, valid {}, test {}'.format(data_set['train']['x'].shape,
                                                            data_set['valid']['x'].shape,
                                                            data_set['test']['x'].shape))

    print('Round %d===============================================================' % i)
    logging.debug('Round %d===============================================================' % i)
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
    i += 1
