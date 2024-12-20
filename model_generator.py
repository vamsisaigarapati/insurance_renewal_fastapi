import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, f1_score, make_scorer,accuracy_score
import sqlite3
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import numpy as np
import os


def fetch_data_into_dataframe(conn):
    query = """
    SELECT 
        fi.Upper_Age,
        fi.Lower_Age,
        fi.Reco_Policy_Premium,
        c.City_Code,
        a.Accomodation_Type,
        fi.Reco_Insurance_Type,
        fi.Is_Spouse,
        h.Health_Indicator,
        fi.Holding_Policy_Duration,
        fi.Holding_Policy_Type,
        fi.class
    FROM 
        FactInsurance fi
    JOIN 
        City c ON fi.City_ID = c.City_ID
    JOIN 
        Accomodation a ON fi.Accomodation_ID = a.Accomodation_ID
    JOIN 
        HealthIndicator h ON fi.Indicator_ID = h.Indicator_ID;
    """
    df = pd.read_sql_query(query, conn)
    return df

db_file = "insurance_data.db"
conn = sqlite3.connect(db_file)
insurance = fetch_data_into_dataframe(conn)
print(insurance.head()) 

conn.close()

# strat_train_set, strat_test_set = train_test_split(housing, test_size=0.20, stratify=insurance["Reco_Policy_Premium_cat"], random_state=42)
X = insurance.drop(columns=["class"])
y = insurance["class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)



# Preprocessing pipeline
num_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="median")),  # Handle missing values
      # Log transformation
    ("scale", StandardScaler()),  # Standard scaling
    ("minmax", MinMaxScaler()),
    ("log", FunctionTransformer(np.log1p, validate=True))# Optional MinMax scaling
])

cat_pipeline = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="most_frequent")),  # Handle missing values
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode
])

preprocessing = make_column_transformer(
    (num_pipeline, make_column_selector(dtype_include=np.number)),  # Numeric columns
    (cat_pipeline, make_column_selector(dtype_include=object))  # Categorical columns
)

random_forest_pipeline = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", RandomForestClassifier(random_state=42))
])

random_forest_pipeline.fit(X_train, y_train)
    
    # Evaluate on the test set
y_pred = random_forest_pipeline.predict(X_test)
orig_f1 = f1_score(y_test, y_pred, average="weighted")
accurtest_facy = accuracy_score(y_test, y_pred)
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print(orig_f1)
import dill
with open('random_forest_pipeline.pkl', 'wb') as f:
    dill.dump(random_forest_pipeline, f)
import dill
with open('random_forest_pipeline.pkl', 'rb') as f:
    reloaded_model = dill.load(f)
rel_f1=f1_score(y_test, reloaded_model.predict(X_test), average="weighted")
print(rel_f1)