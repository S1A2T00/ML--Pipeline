import os                     # Used for file and folder operations
import logging                # Used to create log files for debugging
import string                 # Contains punctuation characters
import pandas as pd           # Used to read and manipulate datasets
import nltk                   # Natural Language Toolkit for NLP

from sklearn.preprocessing import LabelEncoder  # Converts categorical labels into numbers
from nltk.stem import PorterStemmer             # Used for stemming words
from nltk.corpus import stopwords               # Contains English stopwords

# ==========================================================
# Download required NLTK resources (downloads only if missing)
# ==========================================================

resources = [
    "punkt",       # Tokenizer
    "punkt_tab",   # Required in newer NLTK versions
    "stopwords"    # English stopwords
]

for resource in resources:
    try:
        # Check whether the resource already exists
        if resource == "punkt":
            nltk.data.find("tokenizers/punkt")

        elif resource == "punkt_tab":
            nltk.data.find("tokenizers/punkt_tab")

        elif resource == "stopwords":
            nltk.data.find("corpora/stopwords")

    except LookupError:
        # Download resource if not found
        nltk.download(resource)

# ==========================================================
# Create logs directory
# ==========================================================

log_dir = "logs"

# Creates logs folder if it doesn't already exist
os.makedirs(log_dir, exist_ok=True)

# ==========================================================
# Configure Logger
# ==========================================================

# Create logger object
logger = logging.getLogger("data_preprocessing")

# Log every message (DEBUG, INFO, WARNING, ERROR)
logger.setLevel(logging.DEBUG)

# Prevent duplicate log handlers
if not logger.handlers:

    # Print logs in terminal
    console_handler = logging.StreamHandler()

    # Save logs into file
    file_handler = logging.FileHandler(
        os.path.join(log_dir, "data_preprocessing.log")
    )

    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)

    # Format of log message
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ==========================================================
# Load stopwords once (faster than loading every function call)
# ==========================================================

STOP_WORDS = set(stopwords.words("english"))

# Create stemmer object once
STEMMER = PorterStemmer()

# ==========================================================
# Function to clean and preprocess text
# ==========================================================

def transform_text(text):

    # Convert any datatype into string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Split sentence into words (tokenization)
    tokens = nltk.word_tokenize(text)

    # Keep only letters and numbers
    # Removes punctuation like ! @ # $
    tokens = [
        word
        for word in tokens
        if word.isalnum()
    ]

    # Remove stopwords
    # Example:
    # "This is a movie"
    # becomes
    # "movie"

    tokens = [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]

    # Convert words to their root form
    # playing -> play
    # running -> run

    tokens = [
        STEMMER.stem(word)
        for word in tokens
    ]

    # Join words back into one sentence
    return " ".join(tokens)


# ==========================================================
# Function to preprocess complete dataframe
# ==========================================================

def preprocess_df(df, text_column="text", target_column="target"):

    try:

        logger.debug("Starting preprocessing")

        # Create LabelEncoder object
        encoder = LabelEncoder()

        # Convert labels into numeric values
        # Example:
        # spam -> 1
        # ham -> 0

        df[target_column] = encoder.fit_transform(df[target_column])

        logger.debug("Target column encoded")

        # Remove duplicate rows
        df = df.drop_duplicates()

        # Reset row index
        df = df.reset_index(drop=True)

        logger.debug("Duplicates removed")

        # Apply text preprocessing on every row
        df[text_column] = df[text_column].astype(str).apply(transform_text)

        logger.debug("Text transformed successfully")

        return df

    except Exception as e:

        # Save error in log file
        logger.exception("Error during preprocessing")

        raise e


# ==========================================================
# Main Function
# ==========================================================

def main():

    try:

        # Location of raw dataset
        train_path = "./data/raw/train.csv"
        test_path = "./data/raw/test.csv"

        # Read CSV files
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.debug("Data loaded successfully")

        # Clean training data
        train_processed = preprocess_df(train_df)

        # Clean testing data
        test_processed = preprocess_df(test_df)

        # Folder where processed files will be saved
        output_dir = "./data/interim"

        os.makedirs(output_dir, exist_ok=True)

        # Save processed training dataset
        train_processed.to_csv(
            os.path.join(output_dir, "train_processed.csv"),
            index=False
        )

        # Save processed testing dataset
        test_processed.to_csv(
            os.path.join(output_dir, "test_processed.csv"),
            index=False
        )

        logger.debug("Processed files saved successfully.")

    except Exception as e:

        # Save error into log file
        logger.exception("Processing failed")

        # Print error on screen
        print(e)


# ==========================================================
# Program starts here
# ==========================================================

if __name__ == "__main__":

    # Call main function
    main()