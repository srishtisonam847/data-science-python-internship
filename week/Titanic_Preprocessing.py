import pandas as pd

url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/titanic.csv"
df = pd.read_csv(url)

df["Age"] = df.groupby("Pclass")["Age"].transform(lambda s: s.fillna(s.median()))
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
df["Cabin"] = df["Cabin"].fillna("Unknown")
df = df.drop_duplicates()

categorical_cols = ["Sex", "Embarked", "Cabin"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

Q1 = df["Fare"].quantile(0.25)
Q3 = df["Fare"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

fare_outliers = df[(df["Fare"] < lower) | (df["Fare"] > upper)]

print("Final shape:", df.shape)
print("Missing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("Potential Fare outliers:", len(fare_outliers))
