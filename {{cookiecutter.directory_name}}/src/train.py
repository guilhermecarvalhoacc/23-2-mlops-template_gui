# all imports to pandas
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder
from lightgbm import LGBMClassifier


cat_cols = ["job", "marital", "education", "housing"]
num_cols = ["age", "balance", "duration", "campaign"]

df = pd.read_csv("../data/bank_clean.csv")

X = df.drop("deposit", axis=1)
y = df["deposit"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1912)

one_hot_enc = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore", drop="first"),
    cat_cols),
    remainder="passthrough")

X_train = one_hot_enc.fit_transform(X_train)
X_train = pd.DataFrame(X_train, columns=one_hot_enc.get_feature_names_out())

X_train.head(2)

X_test = pd.DataFrame(one_hot_enc.transform(X_test), columns=one_hot_enc.get_feature_names_out())
X_test.head(2)

model = LGBMClassifier()
model.fit(X_train, y_train)

# Specify the file path where you want to save the pickle file
file_path = "../models/model.pkl"

# Save the model as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(model, f)

file_path = "../models/ohe.pkl"

# Save the OneHotEncoder as a pickle file
with open(file_path, "wb") as f:
    pickle.dump(one_hot_enc, f)