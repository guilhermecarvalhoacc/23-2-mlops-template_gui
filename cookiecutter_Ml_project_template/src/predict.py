import pandas as pd
import pickle

c_data = "../data/"
c_models = "../models/"

df = pd.read_csv(c_data + "bank_clean.csv")

df.drop("deposit", axis=1, inplace=True)

model = pickle.load(open(c_models + 'model.pkl', 'rb'))
one_hot_enc = pickle.load(open(c_models + 'ohe.pkl', 'rb'))

X_pred = pd.DataFrame(one_hot_enc.transform(df), columns=one_hot_enc.get_feature_names_out())
y_pred = model.predict(X_pred)

# Convert y_pred to a pandas Series and use .map() to map values
df["y_pred"] = pd.Series(y_pred).map({0: "no", 1: "yes"})

df.to_csv(c_data + "bank_predict.csv", index=False)
