import pandas as pd
import numpy as np
import src.config as config


def drop_unnecesary_id(df: pd.DataFrame) -> pd.DataFrame:
    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    return df


def fill_sex(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df[config.SEX_COL].value_counts().index[0]
    df[config.SEX_COL] = df[config.SEX_COL].fillna(most_freq)
    return df


def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df[config.CAT_COLS] = df[config.CAT_COLS].astype('object')

    ohe_int_cols = df[config.OHE_COLS].select_dtypes('number').columns
    df[ohe_int_cols] = df[ohe_int_cols].astype(np.int8)

    df[config.REAL_COLS] = df[config.REAL_COLS].astype(np.float32)
    return df


def set_idx(df: pd.DataFrame, idx_col: str) -> pd.DataFrame:
    df = df.set_index(idx_col)
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = set_idx(df, config.ID_COL)
    df = drop_unnecesary_id(df)
    df = fill_sex(df)
    df = cast_types(df)
    return df


def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[config.TARGET_COLS] = df[config.TARGET_COLS].astype(np.int8)
    return df


def extract_target(df: pd.DataFrame):
    df, target = df.drop(config.TARGET_COLS, axis=1), df[config.TARGET_COLS]
    return df, target