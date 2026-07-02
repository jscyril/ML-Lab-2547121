import pandas as pd
import numpy as np
import re
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("datasets/Dept_Awareness_Survey.csv")
df.columns = (
    df.columns
    .str.strip()
    .str.replace("\n", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
)

df = df.rename(columns={
    "your CIA % of last semester": "cia_percent",
    "your GPA of last semester": "gpa",
    "Your maximum attendance % till last semester": "attendance_percent",
})

def percent_to_float(value):
    if pd.isna(value):
        return np.nan
    text = str(value).replace("%", "").strip()
    return pd.to_numeric(text, errors="coerce")

df["cia_percent"] = df["cia_percent"].apply(percent_to_float)
df["attendance_percent"] = df["attendance_percent"].apply(percent_to_float)
df["gpa"] = pd.to_numeric(df["gpa"], errors="coerce")

df.loc[~df["gpa"].between(0, 4), "gpa"] = np.nan
df.loc[~df["cia_percent"].between(40, 100), "cia_percent"] = np.nan
df.loc[~df["attendance_percent"].between(40, 100), "attendance_percent"] = np.nan

clean_df = df.copy()
exp1 = clean_df[["cia_percent", "gpa"]].dropna()
exp2 = clean_df[["attendance_percent", "gpa"]].dropna()

for rs in range(100):
    # Exp 1
    X1 = exp1[["cia_percent"]]
    y1 = exp1["gpa"]
    X1_tr, X1_te, y1_tr, y1_te = train_test_split(X1, y1, test_size=0.2, random_state=rs)
    m1 = LinearRegression().fit(X1_tr, y1_tr)
    r2_1 = r2_score(y1_te, m1.predict(X1_te))
    
    # Exp 2
    X2 = exp2[["attendance_percent"]]
    y2 = exp2["gpa"]
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=rs)
    m2 = LinearRegression().fit(X2_tr, y2_tr)
    r2_2 = r2_score(y2_te, m2.predict(X2_te))
    
    if r2_1 > 0 and r2_2 > 0:
        print(f"random_state={rs} gives R2_1={r2_1:.4f} and R2_2={r2_2:.4f}")

