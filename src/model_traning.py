import os
import numpy as np
import pandas as pd
import pickle
import logging
from sklearn.ensemble import RandomForestClassifier

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

        train_data = load_data("./data/processed/train_tfidf.csv")

        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values

        model = train_model(
            X_train,
            y_train,
            params
        )

        save_model(
            model,
            "./models/model.pkl"
        )

        logger.info("Model Building Completed Successfully!")

    except Exception as e:
        logger.error(
            "Failed to complete model building process: %s",
            e
        )
        print("Error:", e)


if __name__ == "__main__":
    main()