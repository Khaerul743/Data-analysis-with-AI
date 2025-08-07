import pandas as pd


def beauty_output(path: str):
    df = pd.read_csv(path)
    return df.to_string()
