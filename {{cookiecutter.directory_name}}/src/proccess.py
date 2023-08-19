import pandas as pd

df = pd.read_csv("../data/bank.csv")
dep_mapping = {"yes": 1, "no": 0}
# Convert the column to category and map the values
df["deposit"] = df["deposit"].astype("category").map(dep_mapping)
df = df.drop(labels = ["default", "contact", "day", "month", "pdays", "previous", "loan", "poutcome", "poutcome"], axis=1)

# saving files
df.to_csv("../data/bank_clean.csv", index=False)    
