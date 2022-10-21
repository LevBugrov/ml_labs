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

def data_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    # заполнение пропусков
    df['Возраст курения'] = np.where(df['Статус Курения'] == 'Никогда не курил(а)', 0, df['Возраст курения'])
    df['Сигарет в день'] = np.where(df['Статус Курения'] == 'Никогда не курил(а)', 0, df['Сигарет в день'])
    df['Сигарет в день'] = np.where(df['Сигарет в день'].isna(), 0, df['Сигарет в день'])
    df['Частота пасс кур'] = np.where(df['Пассивное курение'] == 0, 'Ни разу в день', df['Частота пасс кур'])
    df['Возраст алког'] = np.where(df['Алкоголь'] == 'никогда не употреблял', 0, df['Возраст алког'])

    if 'ID_y' in df.columns:
        df = df.drop('ID_y', axis=1)
    if 'ID' in df.columns:
        df = df.set_index('ID')

    """df = df.drop(df[df['Возраст алког'].isna()].index)
    df = df.drop(df[df['Пол'].isna()].index)
    df = df.drop(df[df['Частота пасс кур'].isna()].index)"""
    
    Ohe_int_columns = [config.OHE_COLS[i] for i in range(len(config.OHE_COLS)) if type(df[config.OHE_COLS[i]][0]) == np.int64]
    df[Ohe_int_columns] = df[Ohe_int_columns].astype(np.int8)
    df[config.CAT_COLS] = df[config.CAT_COLS].astype('category')
    df[config.REAL_COLS] = df[config.REAL_COLS].astype(np.float32)
    
    return df
