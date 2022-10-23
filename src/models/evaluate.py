# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.utils import save_as_pickle
import pandas as pd
import src.config as cfg
import pickle
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import json




@click.command()
@click.argument('input_catboost_model', type=click.Path())
@click.argument('input_sklearn_model', type=click.Path())
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path(exists=True))
@click.argument('output_metrics_filepath', type=click.Path())

#@click.argument('input_val_filepath', type=click.Path())



def main(input_data_filepath, input_target_filepath, input_catboost_model, input_sklearn_model, output_metrics_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('evaluate models')

    train = pd.read_pickle(input_data_filepath)
    target = pd.read_pickle(input_target_filepath)
    #val_index = pd.read_pickle()
    print()

    sklearn_model = pickle.load(open(input_sklearn_model, 'rb'))
    catboost_model = pickle.load(open(input_catboost_model, 'rb'))

    catboost_prediction = catboost_model.predict(train)
    sklearn_prediction = sklearn_model.predict(train)

    precision_score_catboost = precision_score(target, catboost_prediction, average='micro')
    recall_score_catboost = recall_score(target, catboost_prediction, average='micro')
    f1_score_catboost = f1_score(target, catboost_prediction, average='micro')
    roc_auc_score_catboost = roc_auc_score(target, catboost_prediction, average='micro')

    precision_score_sklearn = precision_score(target, sklearn_prediction, average='micro')
    recall_score_sklearn = recall_score(target, sklearn_prediction, average='micro')
    f1_score_sklearn = f1_score(target, sklearn_prediction, average='micro')
    roc_auc_score_sklearn = roc_auc_score(target, sklearn_prediction, average='micro')

    metrics = {

        'precision_catboost': precision_score_catboost,
        'recall_catboost':    recall_score_catboost,
        'f1_catboost':        f1_score_catboost,
        'roc_auc_catboost':   roc_auc_score_catboost,
        
        'precision_sklearn':  precision_score_sklearn,
        'recall_sklearn':     recall_score_sklearn,
        'f1_sklearn':         f1_score_sklearn,
        'roc_auc_sklearn':    roc_auc_score_sklearn

    }
    
    with open(output_metrics_filepath, 'w') as outfile:
        json.dump(metrics, outfile)




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()