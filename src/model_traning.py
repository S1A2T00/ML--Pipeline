import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier
import yaml

# ==========================
# Create logs directory
# ==========================
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

# ==========================
# Logger Configuration
# ==========================
logger = logging.getLogger("model_building")
logger.setLevel(logging.DEBUG)

# Avoid duplicate logs
if not logger.handlers:

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "model_building.log")
    )
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


# ==========================
# Random Forest Parameters
# ==========================
params = {
    "n_estimators": 100,
    "random_state": 42
}


# ==========================
# Load Data
# ==========================
def load_data(file_path: str):

    try:
        df = pd.read_csv(file_path)
        logger.debug(
            "Data loaded from %s with shape %s",
            file_path,
            df.shape
        )
        return df

    except FileNotFoundError as e:
        logger.error(e)
        raise

    except pd.errors.ParserError as e:
        logger.error(e)
        raise

    except Exception as e:
        logger.error(e)
        raise


# ==========================
# Train Model
# ==========================
def train_model(X_train, y_train, params):

    try:

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train must contain same number of samples."
            )

        logger.debug("Initializing Random Forest...")

        model = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            random_state=params["random_state"]
        )

        logger.debug("Training started...")

        model.fit(X_train, y_train)

        logger.debug("Training completed.")

        return model

    except Exception as e:
        logger.error(e)
        raise


# ==========================
# Save Model
# ==========================
def save_model(model, file_path):

    try:

        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as f:
            pickle.dump(model, f)

        logger.debug("Model saved to %s", file_path)

    except Exception as e:
        logger.error(e)
        raise


# ==========================
# Main Function
# ==========================
def main():
    try:
        all_params = load_params('params.yaml') or {}
        # Support multiple possible keys and provide defaults
        mp = (
            all_params.get('model_building') or
            all_params.get('model_traning') or
            all_params.get('model_training')
        ) or {}

        # defaults
        default_params = {"n_estimators": 100, "random_state": 42}

        # Safely coerce types and fall back to defaults on error
        try:
            n_estimators = int(mp.get('n_estimators', default_params['n_estimators']))
        except Exception:
            n_estimators = default_params['n_estimators']

        try:
            random_state = int(mp.get('random_state', default_params['random_state']))
        except Exception:
            random_state = default_params['random_state']

        params = {"n_estimators": n_estimators, "random_state": random_state}
        train_data = load_data('./data/processed/train_tfidf.csv')
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)
        
        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)

    except Exception as e:
        logger.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()