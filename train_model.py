import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

# -----------------------------
# Load Dataset
# -----------------------------
application = pd.read_csv("application_record.csv")
credit = pd.read_csv("credit_record.csv")

print("Application Shape:", application.shape)
print("Credit Shape:", credit.shape)

# -----------------------------
# Create Target Variable
# -----------------------------

def create_label(status):
    if status in ['2', '3', '4', '5']:
        return 1
    else:
        return 0

credit["TARGET"] = credit["STATUS"].astype(str).apply(create_label)

credit = credit.groupby("ID")["TARGET"].max().reset_index()

print("\nCredit after grouping:")
print(credit.head())

# -----------------------------
# Merge Datasets
# -----------------------------

df = application.merge(credit, on="ID")

print("\nMerged Shape:", df.shape)

# -----------------------------
# Drop Unnecessary Columns
# -----------------------------

df.drop(columns=["ID"], inplace=True)

# -----------------------------
# Missing Values
# -----------------------------

cat_cols = df.select_dtypes(include="object").columns
num_cols = df.select_dtypes(exclude="object").columns

cat_imputer = SimpleImputer(strategy="most_frequent")
num_imputer = SimpleImputer(strategy="median")

df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
df[num_cols] = num_imputer.fit_transform(df[num_cols])

# -----------------------------
# Encode Categorical Columns
# -----------------------------

encoders = {}

for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# -----------------------------
# Split Features & Target
# -----------------------------

X = df.drop("TARGET", axis=1)
y = df["TARGET"]

print("\nFeatures:")
print(X.columns)

# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------

lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

lr_acc = accuracy_score(y_test, lr_pred)

print("\nLogistic Regression Accuracy:", lr_acc)

# -----------------------------
# Decision Tree
# -----------------------------

dt = DecisionTreeClassifier(random_state=42)

dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_acc = accuracy_score(y_test, dt_pred)

print("Decision Tree Accuracy:", dt_acc)

# -----------------------------
# Random Forest
# -----------------------------

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_acc)

# -----------------------------
# Best Model
# -----------------------------

scores = {
    "Logistic Regression": lr_acc,
    "Decision Tree": dt_acc,
    "Random Forest": rf_acc
}

best_model = max(scores, key=scores.get)

print("\nBest Model:", best_model)

if best_model == "Logistic Regression":
    model = lr

elif best_model == "Decision Tree":
    model = dt

else:
    model = rf

# -----------------------------
# Save Model
# -----------------------------

joblib.dump(model, "model.pkl")

joblib.dump(encoders, "encoders.pkl")

joblib.dump(list(X.columns), "columns.pkl")

print("\nModel Saved Successfully!")