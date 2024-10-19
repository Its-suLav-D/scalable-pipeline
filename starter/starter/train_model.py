# Script to train machine learning model.

from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
import pandas as pd
from ml.data import process_data
from ml.model import train_model
import logging
import joblib

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add code to load in the data.
try:
    data = pd.read_csv("starter/data/census.csv")
    logging.info(f"Data loaded successfully. Shape: {data.shape}")

    # Clean column names by stripping leading/trailing spaces
    data.columns = data.columns.str.strip()

    logging.info(f"Cleaned columns in the dataset: {data.columns.tolist()}")

except FileNotFoundError:
    logging.error("File 'census.csv' not found. Please check the file path.")
    exit(1)


# Optional enhancement, use K-fold cross validation instead of a train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Process the test data with the process_data function.
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

# Train and save a model.
model = train_model(X_train, y_train)
pd.to_pickle(model, "starter/model/model.pkl")

# Save the encoder and lb.
pd.to_pickle(encoder, "starter/model/encoder.pkl")
pd.to_pickle(lb, "starter/model/lb.pkl")

# Save as .joblib
joblib.dump(model, "starter/model/model.joblib")
joblib.dump(encoder, "starter/model/encoder.joblib")
joblib.dump(lb, "starter/model/lb.joblib")
