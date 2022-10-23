import pandas as pd
import numpy as np
import src.config as cfg

def add_healthy_lifestyle(df: pd.DataFrame) -> pd.DataFrame:
    for sport, pass_kur in zip(df['Спорт, клубы'].values, df['Пассивное курение'].values):
        df["ЗОЖ"] = sport - pass_kur
    
    cfg.OHE_COLS.append("ЗОЖ")
    return df 

def add_susceptibility_to_disease(df: pd.DataFrame) -> pd.DataFrame:
    for a,b,c in zip(df['Хроническое заболевание легких'].values, df['ВИЧ/СПИД'].values, df['Сахарный диабет'].values):
        df["предрасположенность к болезням"] = a+b+c

    
    cfg.OHE_COLS.append("предрасположенность к болезням")
    return df 
