# -*- coding: utf-8 -*-
from cgi import test
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
import pandas as pd
import src.config as cfg
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from catboost import CatBoostClassifier, Pool
#from sklearn.tree import DecisionTreeClassifier
from src.features import features
from sklearn.linear_model import RidgeClassifier
from catboost import Pool, CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_catboost_model', type=click.Path())
@click.argument('input_sklearn_model', type=click.Path())
@click.argument('output_predictions_filepath', type=click.Path())

def main(input_train_filepath, input_target_filepath, output_predictions_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('train models')

    #load data
    train = pd.read_pickle(input_train_filepath)
    target = pd.read_pickle(input_target_filepath)
        
    train[cfg.CAT_COLS] = train[cfg.CAT_COLS].astype('object')
    

    encod = LabelEncoder()
    for i in cfg.OHE_COLS:
            train[i] = encod.fit_transform(train[i])
    for i in cfg.REAL_COLS:
            train[i] = encod.fit_transform(train[i])
    for i in cfg.CAT_COLS:
            train[i] = encod.fit_transform(train[i])
            
    test[cfg.CAT_COLS] = test[cfg.CAT_COLS].astype('object')

    test = features.add_early_wakeup(test)
    encod = LabelEncoder()
    for i in cfg.OHE_COLS:
            test[i] = encod.fit_transform(test[i])
    for i in cfg.REAL_COLS:
            test[i] = encod.fit_transform(test[i])
    for i in cfg.CAT_COLS:
            test[i] = encod.fit_transform(test[i])
    

    train_x, val_x, train_y, val_y = train_test_split(train, target, test_size=0.2, 
                                                      shuffle=True, random_state=cfg.RANDOM_STATE, 
                                                      stratify=target.iloc[:,[1, 2, 3, 4]].sum(axis=1))
    
    #models
    ridge = RidgeClassifier()
    ridge.fit(train_x, train_y)
    y_pred_rc = ridge.predict(val_x)


    cat = CatBoostClassifier(iterations=100, loss_function='MultiLogloss', 
                                    eval_metric='MultiLogloss', learning_rate=0.05, 
                                    bootstrap_type='Bayesian', boost_from_average=False, 
                                    leaf_estimation_iterations=1, leaf_estimation_method='Gradient')
    cat.fit(train_x, train_y)
    y_pred_cb = cat.predict(val_x)


    #output
    predictions = {

        'catboost_prediction': y_pred_cb,
        'sklearn_prediction': y_pred_rc,

    }

    pickle.dump(predictions, open(output_predictions_filepath, 'wb'))




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
