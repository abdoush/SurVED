import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from itertools import permutations
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


class Dataset:
    def __init__(self, dataset_file_path=None, number_of_splits=5, test_fract=0,
                 p='full', events_only=True,  action='censor', drop_feature=None, normalize_target=True,
                 random_seed=20, train_splits_seed=20):
        self.dataset_file_path = dataset_file_path
        self.number_of_splits = number_of_splits
        self.p = p # if events_only=True then p is the events percentage, else it is the percentage to drop from the whole data
        self.normalize_target = normalize_target
        self.events_only = events_only
        self.action = action  # action='censor' or 'drop'
        self.drop_feature = drop_feature
        self.df = self._load_data()
        self.rest_df, self.test_df = self._get_test_split(fract=test_fract, seed=random_seed)
        self.n_splits = self._get_n_splits(seed=train_splits_seed)
        self.input_shape = self.test_df.shape[1] - 2
        self.print_dataset_summery()

    def get_dataset_name(self):
        pass

    def _preprocess_x(self, x_df):
        pass

    def _preprocess_y(self, y_df, normalizing_val=None):
        pass

    def _preprocess_e(self, e_df):
        pass

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        pass

    def _load_data(self):
        pass

    def get_x_dim(self):
        return self.df.shape[1]-2

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        if (x_test_df is not None) & (x_tune_df is not None):
            return x_train_df.values, x_val_df.values, x_test_df.values, x_tune_df.values
        elif x_test_df is not None:
            return x_train_df.values, x_val_df.values, x_test_df.values
        else:
            return x_train_df.values, x_val_df.values

    def print_dataset_summery(self):
        s = 'Dataset Description =======================\n'
        s += 'Dataset Name: {}\n'.format(self.get_dataset_name())
        s += 'Dataset Shape: {}\n'.format(self.df.shape)
        s += 'Events: %.2f %%\n' % (self.df['E'].sum()*100 / len(self.df))
        s += 'NaN Values: %.2f %%\n' % (self.df.isnull().sum().sum()*100 / self.df.size)
        s += 'Size and Events % in splits: '
        for split in self.n_splits:
            s += '({}, {:.2f}%), '.format((split.shape[0]), (split["E"].mean()*100))
        s += '\n'
        if self.test_df is not None:
            s += '-------------------------------------------\n'
            s += 'Hold-out Testset % of Data: {:.2f}%\n'.format((self.test_df.shape[0] * 100 / self.df.shape[0]))
            s += 'Hold-out Testset Size and Events %: ({:}, {:.2f}%) \n'.format(self.test_df.shape[0], (self.test_df["E"].mean()*100))
        s += '===========================================\n'
        print(s)
        return s

    @staticmethod
    def max_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = ((df_transformed[col]) / df_transformed[col].max()) ** powr
        return df_transformed

    @staticmethod
    def log_transform(df, cols):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = np.abs(np.log(df_transformed[col] + 1e-8))
        return df_transformed

    @staticmethod
    def power_transform(df, cols, powr):
        df_transformed = df.copy()
        for col in cols:
            df_transformed[col] = df_transformed[col] ** powr
        return df_transformed

    def _get_test_split(self, fract=0.4, seed=20):
        if fract == 0:
            return self.df, None
        rest_df, test_df = train_test_split(self.df, test_size=fract, random_state=seed, shuffle=True, stratify=self.df['E'])
        return rest_df, test_df

    def _get_n_splits(self, seed=20):
        k = self.number_of_splits
        train_df = self.rest_df
        df_splits = []
        for i in range(k, 1, -1):
            train_df, test_df = train_test_split(train_df, test_size=(1 / i), random_state=seed, shuffle=True,
                                                 stratify=train_df['E'])
            df_splits.append(test_df)
            if i == 2:
                df_splits.append(train_df)
        return df_splits

    def get_train_val_test_final_eval(self, val_id):
        if self.test_df is None:
            print('No hold-out test set found')
            return
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = self.test_df
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_test_from_splits_for_date_final_eval(self, val_id):
        # df_splits_temp = self.n_splits.copy()
        # val_df = df_splits_temp[val_id]
        # test_df = df_splits_temp[test_id]
        # train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        # train_df = pd.concat(train_df_splits)

        if self.test_df is None:
            print('No hold-out test set found')
            return
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = self.test_df
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)


        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        end_time = max(y_train)

        preprocessed = {
            'train': self._formatted_data(x=x_train, t=y_train, e=e_train),
            'test': self._formatted_data(x=x_test, t=y_test, e=e_test),
            'valid': self._formatted_data(x=x_val, t=y_val, e=e_val),
            'end_t': end_time,
            'covariates': self.covariates,
            'one_hot_indices': [], #encoded_indices,
            'imputation_values': [] #imputation_values
        }

        return preprocessed


    def get_train_val_test_from_splits(self, val_id, test_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_test_from_splits_for_date(self, val_id, test_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        end_time = max(y_train)

        preprocessed = {
            'train': self._formatted_data(x=x_train, t=y_train, e=e_train),
            'test': self._formatted_data(x=x_test, t=y_test, e=e_test),
            'valid': self._formatted_data(x=x_val, t=y_val, e=e_val),
            'end_t': end_time,
            'covariates': self.covariates,
            'one_hot_indices': [], #encoded_indices,
            'imputation_values': [] #imputation_values
        }

        return preprocessed


    def get_val_test_train_exclude_one_from_splits(self, val_id, test_id, excluded_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id, excluded_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_val_test_sampled_train_from_splits(self, val_id, test_id, frac=0.8, replace=True, seed=20):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id]]
        train_df = pd.concat(train_df_splits)

        sampled_train_df = train_df.sample(frac=frac, replace=replace, random_state=seed)

        x_train_df, y_train_df, e_train_df = self._split_columns(sampled_train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df)

        x_train, x_val, x_test = self._preprocess_x(x_train_df), \
                                 self._preprocess_x(x_val_df), \
                                 self._preprocess_x(x_test_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                 self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test = self._preprocess_e(e_train_df), \
                                 self._preprocess_e(e_val_df), \
                                 self._preprocess_e(e_test_df)

        ye_train, ye_val, ye_test = np.array(list(zip(y_train, e_train))), \
                                    np.array(list(zip(y_val, e_val))), \
                                    np.array(list(zip(y_test, e_test)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test)

    def get_train_val_test_tune_from_splits(self, val_id, test_id, tune_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        tune_df = df_splits_temp[tune_id]
        test_df = df_splits_temp[test_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id, test_id, tune_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)
        x_test_df, y_test_df, e_test_df = self._split_columns(test_df)
        x_tune_df, y_tune_df, e_tune_df = self._split_columns(tune_df)

        self._fill_missing_values(x_train_df, x_val_df, x_test_df, x_tune_df)

        x_train, x_val, x_test, x_tune = self._preprocess_x(x_train_df), \
                                         self._preprocess_x(x_val_df), \
                                         self._preprocess_x(x_test_df), \
                                         self._preprocess_x(x_tune_df)

        x_train, x_val, x_test = self._scale_x(x_train, x_val, x_test)

        y_normalizing_val = y_train_df.max()

        y_train, y_val, y_test, y_tune = self._preprocess_y(y_train_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_val_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_test_df, normalizing_val=y_normalizing_val), \
                                         self._preprocess_y(y_tune_df, normalizing_val=y_normalizing_val)

        e_train, e_val, e_test, e_tune = self._preprocess_e(e_train_df), \
                                         self._preprocess_e(e_val_df), \
                                         self._preprocess_e(e_test_df), \
                                         self._preprocess_e(e_tune_df)

        ye_train, ye_val, ye_test, ye_tune = np.array(list(zip(y_train, e_train))), \
                                             np.array(list(zip(y_val, e_val))), \
                                             np.array(list(zip(y_test, e_test))), \
                                             np.array(list(zip(y_tune, e_tune)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val,
                x_test, ye_test, y_test, e_test,
                x_tune, ye_tune, y_tune, e_tune)


    @staticmethod
    def get_shuffled_pairs(x, y, e, seed=None):
        x_sh, y_sh, e_sh = shuffle(x, y, e, random_state=seed)
        y_diff = y_sh - y
        fltr = (e == 1) & (y_diff > 0)  # choose the first item in the pair to be an event
        return x[fltr], y[fltr], x_sh[fltr], y_sh[fltr], y_diff[fltr]

    def get_train_val_from_splits(self, val_id):
        df_splits_temp = self.n_splits.copy()
        val_df = df_splits_temp[val_id]
        train_df_splits = [df_splits_temp[i] for i in range(len(df_splits_temp)) if i not in [val_id]]
        train_df = pd.concat(train_df_splits)

        x_train_df, y_train_df, e_train_df = self._split_columns(train_df)
        x_val_df, y_val_df, e_val_df = self._split_columns(val_df)

        self._fill_missing_values(x_train_df, x_val_df)

        x_train, x_val = self._preprocess_x(x_train_df), self._preprocess_x(x_val_df)

        x_train, x_val = self._scale_x(x_train, x_val)

        y_train, y_val = self._preprocess_y(y_train_df), self._preprocess_y(y_val_df)

        e_train, e_val = self._preprocess_e(e_train_df), self._preprocess_e(e_val_df)

        ye_train, ye_val = np.array(list(zip(y_train, e_train))), np.array(list(zip(y_val, e_val)))

        return (x_train, ye_train, y_train, e_train,
                x_val, ye_val, y_val, e_val)

    def _formatted_data(self, x, t, e):
        survival_data = {'x': x, 't': t, 'e': e}
        return survival_data

    @staticmethod
    def _split_columns(df):
        y_df = df['T']
        e_df = df['E']
        x_df = df.drop(['T', 'E'], axis=1)
        return x_df, y_df, e_df

    def test_dataset(self):
        combs = list(permutations(range(self.number_of_splits), 2))
        for i, j in combs:
            (x_train, ye_train, y_train, e_train,
             x_val, ye_val, y_val, e_val,
             x_test, ye_test, y_test, e_test) = self.get_train_val_test_from_splits(i, j)
            assert np.isnan(x_train).sum() == 0
            assert np.isnan(x_val).sum() == 0
            assert np.isnan(x_test).sum() == 0


class Metabric(Dataset):
    def _load_data(self):
        base_file_name = '.'.join(self.dataset_file_path.split('.')[:-1])
        xdf = pd.read_csv(base_file_name+'.csv')
        ydf = pd.read_csv(base_file_name+'_label.csv')
        ydf.rename(columns={'event_time': 'T', 'label': 'E'}, inplace=True)
        df = xdf.join(ydf)
        self.covariates = [x for x in df.columns if x not in ['T', 'E']]
        return df

    def get_dataset_name(self):
        return 'METABRIC'

    def _preprocess_x(self, x_df):
        return x_df

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

    def _preprocess_y(self, y_df, normalizing_val=None):
        if self.normalize_target:
            if normalizing_val is None:
                normalizing_val = y_df.max()
            return ((y_df / normalizing_val).values ** 0.5).astype('float32')
        else:
            return y_df.values.astype('float32')


    def _preprocess_e(self, e_df):
        return e_df.values.astype('float32')


class Flchain(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col='idx')
        df['sex'] = df['sex'].map(lambda x: 0 if x == 'M' else 1)
        df.drop('chapter', axis=1, inplace=True)
        df['sample.yr'] = df['sample.yr'].astype('category')
        df['flc.grp'] = df['flc.grp'].astype('category')
        df.rename(columns={'futime': 'T', 'death': 'E'}, inplace=True)
        self.covariates = [x for x in df.columns if x not in ['T', 'E']]
        ohdf = pd.get_dummies(df)
        return ohdf

    def get_dataset_name(self):
        return 'flchain'

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        m = x_train_df['creatinine'].median()
        x_train_df['creatinine'].fillna(m, inplace=True)
        x_val_df['creatinine'].fillna(m, inplace=True)
        if x_test_df is not None:
            x_test_df['creatinine'].fillna(m, inplace=True)
        if x_tune_df is not None:
            x_tune_df['creatinine'].fillna(m, inplace=True)

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if self.normalize_target:
            if normalizing_val is None:
                normalizing_val = y_df.max()
            return ((y_df / normalizing_val).values ** 0.5).astype('float32')
        else:
            return ((y_df).values).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.values.astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class Nwtco(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, )
        df.drop(columns=['idx'], inplace=True)
        df = (df.assign(instit_2=df['instit'] - 1,
                        histol_2=df['histol'] - 1,
                        study_4=df['study'] - 3,
                        stage=df['stage'].astype('category'))
              .drop(['seqno', 'instit', 'histol', 'study'], axis=1))
        for col in df.columns.drop('stage'):
            df[col] = df[col].astype('float32')
        df.rename(columns={'edrel': 'T', 'rel': 'E'}, inplace=True)
        self.covariates = [x for x in df.columns if x not in ['T', 'E']]
        ohdf = pd.get_dummies(df)
        return ohdf

    def get_dataset_name(self):
        return 'Nwtco'

    def _preprocess_x(self, x_df):
        return x_df

    def _preprocess_y(self, y_df, normalizing_val=None):
        if self.normalize_target:
            if normalizing_val is None:
                normalizing_val = y_df.max()
            return ((y_df / normalizing_val).values ** 0.5).astype('float32')
        else:
            return ((y_df).values).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.values.astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val


class Support(Dataset):
    def _load_data(self):
        df = pd.read_csv(self.dataset_file_path, index_col=0)
        one_hot_encoder_list = ['sex', 'dzgroup', 'dzclass', 'income', 'race', 'ca', 'sfdm2']
        if self.drop_feature is None:
            to_drop = ['hospdead', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'aps', 'sps', 'surv2m', 'surv6m', 'totmcst']
        else:
            to_drop = ['hospdead', 'prg2m', 'prg6m', 'dnr', 'dnrday', 'aps', 'sps', 'surv2m', 'surv6m', 'totmcst'] + [self.drop_feature]
            if self.drop_feature in one_hot_encoder_list:
                one_hot_encoder_list.remove(self.drop_feature)

        df.drop(columns=to_drop, inplace=True)
        df.rename(columns={'d.time': 'T', 'death': 'E'}, inplace=True)
        self.covariates = [x for x in df.columns if x not in ['T', 'E']]
        ohdf = pd.get_dummies(df, prefix=one_hot_encoder_list, columns=one_hot_encoder_list)
        self.encoded_indices = self.one_hot_indices(ohdf, one_hot_encoder_list)

        if self.p != 'full':
            if self.events_only:
                number_of_events = ohdf['E'].sum()
                N = ohdf['E'].count()
                number_of_censored = N - number_of_events
                if self.action == 'drop':
                    drop_n = number_of_events - int(number_of_censored * self.p /(1 - self.p))
                    idxs = list(ohdf[(ohdf['E'] == 1)].sample(drop_n, random_state=20).index)
                    ohdf = ohdf[~ohdf.index.isin(idxs)]
                else:
                    censor_n = number_of_events - int(self.p * N)
                    idxs = list(ohdf[(ohdf['E'] == 1)].sample(censor_n, random_state=20).index)
                    np.random.seed(20)
                    f = np.random.uniform(0, 1, len(idxs))
                    ohdf.loc[ohdf.index.isin(idxs), 'T'] -= f * ohdf.loc[ohdf.index.isin(idxs), 'T']
                    ohdf.loc[ohdf.index.isin(idxs), 'E'] = 0
            else:
                number_of_samples = ohdf.shape[0]
                drop_n = int(self.p * number_of_samples)
                idxs = list(ohdf.sample(drop_n, random_state=20).index)
                ohdf = ohdf[~ohdf.index.isin(idxs)]
        return ohdf

    def get_dataset_name(self):
        return 'support'

    def _preprocess_x(self, x_df):
        features = ['totcst', 'charges', 'pafi', 'sod']
        if ~(self.drop_feature is None):
            if self.drop_feature in features:
                features.remove(self.drop_feature)
        return super().log_transform(x_df, features)

    def _preprocess_y(self, y_df, normalizing_val=None):
        if self.normalize_target:
            if normalizing_val is None:
                normalizing_val = y_df.max()
            return ((y_df / normalizing_val).values ** 0.1).astype('float32')
        else:
            return ((y_df).values).astype('float32')

    def _preprocess_e(self, e_df):
        return e_df.values.astype('float32')

    def _scale_x(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        scaler = StandardScaler().fit(x_train_df)
        x_train = scaler.transform(x_train_df)
        x_val = scaler.transform(x_val_df)
        if (x_tune_df is not None) & (x_test_df is not None):
            x_test = scaler.transform(x_test_df)
            x_tune = scaler.transform(x_tune_df)
            return x_train, x_val, x_test, x_tune
        elif x_test_df is not None:
            x_test = scaler.transform(x_test_df)
            return x_train, x_val, x_test
        else:
            return x_train, x_val

    def _fill_missing_values(self, x_train_df, x_val_df, x_test_df=None, x_tune_df=None):
        imputation_values = self.get_train_median_mode(x=x_train_df.values, categorial=self.encoded_indices)
        imputation_vals_dict = dict(zip(self.df.columns, imputation_values))
        x_train_df.fillna(imputation_vals_dict, inplace=True)
        x_val_df.fillna(imputation_vals_dict, inplace=True)
        if x_test_df is not None:
            x_test_df.fillna(imputation_vals_dict, inplace=True)
        if x_tune_df is not None:
            x_tune_df.fillna(imputation_vals_dict, inplace=True)

    @staticmethod
    def one_hot_indices(dataset, one_hot_encoder_list):
        """
        The function is copied from: https://github.com/paidamoyo/adversarial_time_to_event
        """
        indices_by_category = []
        for colunm in one_hot_encoder_list:
            values = dataset.filter(regex="{}_.*".format(colunm)).columns.values
            indices_one_hot = []
            for value in values:
                indice = dataset.columns.get_loc(value)
                indices_one_hot.append(indice)
            indices_by_category.append(indices_one_hot)
        return indices_by_category

    @staticmethod
    def get_train_median_mode(x, categorial):
        """
        The function is copied from: https://github.com/paidamoyo/adversarial_time_to_event
        """
        def flatten_nested(list_of_lists):
            flattened = [val for sublist in list_of_lists for val in sublist]
            return flattened

        categorical_flat = flatten_nested(categorial)
        imputation_values = []
        median = np.nanmedian(x, axis=0)
        mode = []
        for idx in np.arange(x.shape[1]):
            a = x[:, idx]
            (_, idx, counts) = np.unique(a, return_index=True, return_counts=True)
            index = idx[np.argmax(counts)]
            mode_idx = a[index]
            mode.append(mode_idx)
        for i in np.arange(x.shape[1]):
            if i in categorical_flat:
                imputation_values.append(mode[i])
            else:
                imputation_values.append(median[i])
        return imputation_values


