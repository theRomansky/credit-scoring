import joblib
import pandas as pd
import glob

from sklearn.preprocessing import OneHotEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from catboost import CatBoostClassifier
from sklearn.pipeline import Pipeline

target = pd.read_csv('../data/train_target.csv')


def extraction(data_file):
    extracted_data = pd.read_parquet(data_file)
    extracted_data = extracted_data.drop('rn', axis=1)

    return extracted_data


def generate_new_features(data):
    data['total_delinquencies'] = data[['is_zero_loans5', 'is_zero_loans530', 'is_zero_loans3060', 'is_zero_loans6090', 'is_zero_loans90']].sum(axis=1)

    data['total_undefined_days'] = data[['pclose_flag', 'fclose_flag']].sum(axis=1)

    return data


def encoding(dataset):
    id_column = dataset['id']
    ohe = OneHotEncoder(sparse=False)
    encoded_data = ohe.fit_transform(dataset.drop('id', axis=1))
    df_encoded = pd.DataFrame(data=encoded_data, columns=ohe.get_feature_names_out(),)
    df_encoded = pd.concat([id_column, df_encoded], axis=1).reset_index(drop=True)

    return df_encoded


def group_aggregate(dataset, function):
    agg_data = dataset.groupby('id').agg(function).reset_index()

    return agg_data


def process_data_chunk(data_chunk, existing_data=None):
    extracted_data = extraction(data_chunk)
    new_features_data = generate_new_features(extracted_data)
    encoded_data = encoding(new_features_data)
    aggregated_data = group_aggregate(encoded_data, function='sum')

    if existing_data is None:
        return aggregated_data
    else:
        combined_data = pd.concat([existing_data, aggregated_data], join='outer', ignore_index=True, sort=False)
        combined_data = combined_data.fillna(0)

    return combined_data


existing_data = None
data_files = glob.glob('../data/train_data/train_data_*.pq')

for data_file in data_files:
    existing_data = process_data_chunk(data_file, existing_data)

df = pd.merge(existing_data, target, how='inner', on='id')


class RemoveIDColumn(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=['id', 'flag'], errors='ignore')

catboost_pipeline = Pipeline([
    ('remove_id', RemoveIDColumn()),
    ('classifier', CatBoostClassifier(auto_class_weights='Balanced', depth=7, l2_leaf_reg=0.8))
])


catboost_pipeline.fit(df, target['flag'])


joblib.dump(catboost_pipeline, '../models/catboost_pipeline.pkl')
