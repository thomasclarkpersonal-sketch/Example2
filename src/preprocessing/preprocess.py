import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import joblib

# ---------------------------------------------------------
#portable path to the dataset (please dont touch - Hassan)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # folder of this script
DATA_PATH = os.path.join(BASE_DIR, "..", "..", "data", "raw", "adult.data")

# Load dataset
df = pd.read_csv(DATA_PATH, header=None)

df.columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income"
]

# Replace missing values
df.replace(" ?", None, inplace=True)
df.dropna(inplace=True)

# Encode target
df['income'] = df['income'].map({' <=50K': 0, ' >50K': 1})

# Split features/target
X = df.drop("income", axis=1)
y = df["income"]

# Identify columns
categorical = X.select_dtypes(include=['object']).columns.tolist()
numerical = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

#preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
    ]
)

#save pipeline next to this script
PIPELINE_PATH = os.path.join(BASE_DIR, "preprocessor.pkl")
joblib.dump(preprocessor, PIPELINE_PATH)

print("Preprocessing pipeline saved to:", PIPELINE_PATH)

#save data splits
DATA_SPLITS_PATH = os.path.join(BASE_DIR, "data_splits.pkl")
joblib.dump((X_train, X_test, y_train, y_test), DATA_SPLITS_PATH)

print("Data splits saved to:", DATA_SPLITS_PATH)