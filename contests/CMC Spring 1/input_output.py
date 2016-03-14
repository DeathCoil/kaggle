import pandas as pd
import numpy as np
import datetime as datetime


def load_data():
    train = pd.read_csv('input/train__.csv')
    test = pd.read_csv('input/test__.csv')

    target = train['radiant_win'].values
    test_index = test['match_id'].values
    train.drop(['duration', 'radiant_win', 'tower_status_radiant', 'tower_status_dire',
                'barracks_status_radiant', 'barracks_status_dire'], axis=1, inplace=True)

    return train, test, target, test_index

def save_result(index, result):
    df = pd.DataFrame({"match_id": index, "radiant_win": result})
    df.to_csv("output/" + str(datetime.datetime.now()), float_format="%0.7f", index=False)
