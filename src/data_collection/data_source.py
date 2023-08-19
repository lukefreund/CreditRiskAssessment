import pandas as pd


def load_data():
    filepath = 'data/german_credit_data.csv'
    data = pd.read_csv(filepath)
    return data
