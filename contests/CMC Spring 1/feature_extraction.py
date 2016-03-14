import numpy as np
import pandas as pd
import sklearn
import sklearn.preprocessing


def drop_features(data):
    columns = ["r1_hero", "r2_hero", "r3_hero", "r4_hero", "r5_hero",
               "d1_hero", "d2_hero", "d3_hero", "d4_hero", "d5_hero",
               "lobby_type",
               "radiant_24_count", "radiant_30_count",
               "radiant_32_count", "radiant_33_count", "radiant_35_count", "radiant_37_count",
               "dire_24_count", "dire_30_count",
               "dire_32_count", "dire_33_count", "dire_35_count", "dire_37_count",
               "bans",
              ]
    data.drop(columns, axis=1, inplace=True)
    return data


def add_diff_teams(data):
    for feature_name in ["level", "xp", "gold", "lh", "kills", "deaths", "items"]:
        for func in ["max", "min", "mean", "median"]:
            data["diff_" + feature_name + "_" + func] = data["r_" + feature_name + "_" + func] - \
                                                        data["d_" + feature_name + "_" + func]
    return data



def add_features(data):
    data = sort_players(data)

    data = add_missing_indicator(data)
    data = add_team_statistics(data)
    data = add_diff_teams(data)
    data = add_lobby_type(data)

    return data


def add_lobby_type(data):
    data["lobby_type_0"] = data["lobby_type"] == 0
    data["lobby_type_1"] = data["lobby_type"] == 1
    data["lobby_type_7"] = data["lobby_type"] == 7

    return data


def add_missing_indicator(data):
    columns = ['first_blood_time', 'first_blood_player2',
               'radiant_bottle_time','radiant_courier_time', 'radiant_flying_courier_time',
               'radiant_first_ward_time',
               'dire_bottle_time', 'dire_courier_time', 'dire_flying_courier_time',
               'dire_first_ward_time']

    for column in columns:
        data["isnan" + column] = np.isnan(data[column])


    return data


def add_team_statistics(data):
    for team in ["d", "r"]:
        for feature_name in ["level", "xp", "gold", "lh", "kills", "deaths", "items"]:
            columns = []
            for player in ["1", "2", "3", "4", "5"]:
                columns += [team + player + "_" + feature_name]

            data[team+"_"+feature_name+"_max"] = np.max(data[columns], axis=1)
            data[team+"_"+feature_name+"_min"] = np.min(data[columns], axis=1)
            data[team+"_"+feature_name+"_mean"] = np.mean(data[columns], axis=1)
            data[team+"_"+feature_name+"_mean^3"] = np.mean(data[columns], axis=1) ** 3
            data[team+"_"+feature_name+"_median"] = np.median(data[columns], axis=1)

    return data


def sort_players(data):
    sort_feature = "lh"

    for team in ["d", "r"]:
        sort_by_columns = []
        for player in ["1", "2", "3", "4", "5"]:
            sort_by_columns += [team + player + "_" + sort_feature]
        vals = data[sort_by_columns].values
        sort_idx = list(np.ogrid[[slice(x) for x in vals.shape]])
        sort_idx[1] = vals.argsort(axis=1)

        for feature_name in ["level", "xp", "gold", "lh", "kills", "deaths", "items"]:
            columns = []
            for player in ["1", "2", "3", "4", "5"]:
                columns += [team + player + "_" + feature_name]

            data[columns] = data[columns].values[sort_idx]


    return data




def input_missing(train, test):

    imp = sklearn.preprocessing.Imputer()
    train = imp.fit_transform(train)
    test = imp.transform(test)

    return train, test


def scale_data(train, test):
    scaler = sklearn.preprocessing.StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    return train, test


def shuffle_train(train, target):
    idx = np.arange(len(train))
    np.random.seed(1234)
    np.random.shuffle(idx)
    train = train.iloc[idx]
    target = target[idx]

    return train, target


def get_picks(data):
    N_heroes = 113
    X_pick = np.zeros((data.shape[0], N_heroes))

    for i, match_id in enumerate(data.index):
        for p in range(5):
            X_pick[i, data.ix[match_id, 'r%d_hero' % (p+1)]-1] = 1
            X_pick[i, data.ix[match_id, 'd%d_hero' % (p+1)]-1] = -1

    return X_pick


def add_hero_pairs(train, test, picks_train, picks_test, target):
    N_heroes = 113
    for i in range(N_heroes):
        for j in range(i+1, N_heroes):
            slide = target[np.logical_or(np.logical_and(picks_train[:, i] == 1, picks_train[:, j] == -1),
                                         np.logical_and(picks_train[:, i] == -1, picks_train[:, j] == 1))]
            if len(slide) > 3000:
                train_feature = np.logical_and(picks_train[:, i] == 1, picks_train[:, j] == -1).astype(np.float) - \
                                np.logical_and(picks_train[:, i] == -1, picks_train[:, j] == 1).astype(np.float)
                test_feature = np.logical_and(picks_test[:, i] == 1, picks_test[:, j] == -1).astype(np.float) - \
                               np.logical_and(picks_test[:, i] == -1, picks_test[:, j] == 1).astype(np.float)
                train["d_heroes_"+str(i) + "_" + str(j)] = train_feature
                test["d_heroes_"+str(i) + "_" + str(j)] = test_feature


    return train, test


def calc_rating(data, target):
    N = 113 # heroes

    # calculate each hero-pair synergy and antisynergy
    synergy = np.zeros((N,N))     # sum of wins in matches played together
    antisynergy = np.zeros((N,N)) # sum of wins when played against
    matchcounts = np.zeros((N,N)) # count of matches played together
    matchcounta = np.zeros((N,N)) # count of matches played against

    for match_counter, match_id in enumerate(data.index):
        #synergy when both heroes in win team
        winteam = 'r' if target[match_counter] == 1 else 'd'
        looseteam = 'd' if winteam =='r' else 'r'
        pind     = [0] *5 #player indexes
        antipind = [0] *5 # looser indicies
        # get indexes of players in each tem
        for i in range(5):
            pind[i] = data.ix[match_id, winteam+'%d_hero'%(i+1)]-1
        for i in range(5):
            antipind[i] = data.ix[match_id, looseteam+'%d_hero'%(i+1)]-1
        # accumulate synergy of pairs
        for i in range(5):
            for j in range(i+1,5):
                synergy[pind[i], pind[j]] +=1
                synergy[pind[j], pind[i]] +=1
        # accumulate match counter for playing together
        for i in range(5):
            for j in range(5):
               matchcounts[pind[i], pind[j]] +=1 #together and win
               matchcounts[antipind[i], antipind[j]] +=1 # together and loose

        #antisynergy when hero i in winteam while hero j in loose team
        for i in range(5):
            for j in range(5):
                antisynergy[pind[i], antipind[j]] +=1
                matchcounta[pind[i], antipind[j]] +=1
                matchcounta[antipind[j], pind[i]] +=1

    # normalize
    synergyrate = np.zeros((N,N))
    antisynergyrate = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
          if matchcounts[i,j] !=0:
            synergyrate[i,j] = synergy[i,j]/matchcounts[i,j]
          else:
            synergyrate[i,j] = 0.5
          if matchcounta[i,j] !=0:
            antisynergyrate[i,j] = antisynergy[i,j]/ matchcounta[i,j]
          else:
            antisynergyrate[i,j] = 0.5

    return synergyrate, antisynergyrate


def add_hero_synergy_part(data, synergyrate, antisynergyrate):
    syn1 = np.zeros(len(data))
    syn2 = np.zeros(len(data))
    syn3 = np.zeros(len(data))
    antisyn1 = np.zeros(len(data))
    antisyn2 = np.zeros(len(data))
    antisyn3 = np.zeros(len(data))

    for player1 in range(1, 6):
        for player2 in range(player1+1, 6):
                syn1 += synergyrate[data["r" + str(player1) + "_hero"]-1, data["r" + str(player2) + "_hero"]-1]
                syn2 += synergyrate[data["d" + str(player1) + "_hero"]-1, data["d" + str(player2) + "_hero"]-1]
    syn3 = syn1 - syn2

    for player1 in range(1, 6):
        for player2 in range(1, 6):
                antisyn1 += antisynergyrate[data["r" + str(player1) + "_hero"]-1, data["d" + str(player2) + "_hero"]-1]
                antisyn2 += antisynergyrate[data["d" + str(player1) + "_hero"]-1, data["r" + str(player2) + "_hero"]-1]
    antisyn3 = antisyn1 - antisyn2

    return syn1, syn2, syn3, antisyn1, antisyn2, antisyn3


def add_hero_synergy(train, test, target):
    N = 10
    syn_antisyn_train = np.zeros((6, len(train)))
    syn_antisyn_test = np.zeros((6, len(test)))
    temp_syn_antisyn = np.empty_like(syn_antisyn_test)

    cv = sklearn.cross_validation.KFold(len(train), n_folds=N, shuffle=True, random_state=1234)
    for train_index, test_index in cv:
        synergyrate, antisynergyrate = calc_rating(train.iloc[train_index, :], target[train_index])
        syn_antisyn_train[0][test_index], syn_antisyn_train[1][test_index], syn_antisyn_train[2][test_index], \
        syn_antisyn_train[3][test_index], syn_antisyn_train[4][test_index], syn_antisyn_train[5][test_index] = add_hero_synergy_part(train.iloc[test_index, :], synergyrate, antisynergyrate)
        temp_syn_antisyn = np.array(add_hero_synergy_part(test, synergyrate, antisynergyrate))
        syn_antisyn_test += temp_syn_antisyn/N

    train["synergy1"] = syn_antisyn_train[0]
    train["synergy2"] = syn_antisyn_train[1]
    train["synergy3"] = syn_antisyn_train[2]
    train["antisynergy1"] = syn_antisyn_train[3]
    train["antisynergy2"] = syn_antisyn_train[4]
    train["antisynergy3"] = syn_antisyn_train[5]

    test["synergy1"] = syn_antisyn_test[0]
    test["synergy2"] = syn_antisyn_test[1]
    test["synergy3"] = syn_antisyn_test[2]
    test["antisynergy1"] = syn_antisyn_test[3]
    test["antisynergy2"] = syn_antisyn_test[4]
    test["antisynergy3"] = syn_antisyn_test[5]

    return train, test


def preprocess_data(train, test, target):
    train, target = shuffle_train(train, target)

    train_picks, test_picks = get_picks(train), get_picks(test)
    train, test = add_hero_pairs(train, test, train_picks, test_picks, target)
    train, test = add_hero_synergy(train, test, target)

    train = add_features(train)
    test = add_features(test)

    train = drop_features(train)
    test = drop_features(test)

    train, test = input_missing(train, test)

    train = np.column_stack((train, train_picks))
    test = np.column_stack((test, test_picks))

    train, test = scale_data(train, test)

    return train, test, target