import pickle

names = ["attr_df", "brand_df", "brands", "color_df", "color_name", "color_set", "df_test", "df_train"]

for name in names:
    print(name)
    data = pickle.load("cache/" + name, "r")
    pickle.dump(data, open("cache/" + name, "w"), protocol=0)
