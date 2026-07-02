# 🚀 End-to-End Machine Learning Pipeline using DVC & AWS S3

An end-to-end Machine Learning pipeline that demonstrates how to build a reproducible ML workflow using **Python**, **DVC (Data Version Control)**, and **AWS S3** for remote storage of datasets and model artifacts.

## 📌 Project Overview

This project follows an industry-style Machine Learning workflow, where every stage of the pipeline is modular and version-controlled.

The pipeline includes:

* Data Ingestion
* Data Validation
* Data Transformation
* Feature Engineering
* Model Training
* Model Evaluation
* Data & Model Versioning using DVC
* Remote Storage using AWS S3
* Logging and Exception Handling

---

## 🏗️ Project Structure

```text
ML--Pipeline/
│
├── config/
├── data/
├── logs/
├── notebook/
├── src/
├── artifacts/
├── dvc.yaml
├── params.yaml
├── requirements.txt
├── setup.py
├── main.py
├── README.md
└── .gitignore
```

---

## ⚙️ Technologies Used

* Python 3.x
* Pandas
* NumPy
* Scikit-learn
* DVC
* AWS S3
* Git & GitHub
* YAML
* Pickle
* Logging

---

## 🔄 Machine Learning Pipeline

```text
Data Collection
       │
       ▼
Data Validation
       │
       ▼
Data Transformation
       │
       ▼
Feature Engineering
       │
       ▼
Model Training
       │
       ▼
Model Evaluation
       │
       ▼
Version Model with DVC
       │
       ▼
Store Artifacts in AWS S3
```

---

## 📂 DVC Workflow

```bash
# Initialize DVC
dvc init

# Track dataset
dvc add data/

# Configure AWS S3 as remote storage
dvc remote add -d storage s3://your-bucket-name

# Push data and models to S3
dvc push

# Pull data and models
dvc pull

# Reproduce the pipeline
dvc repro
```

---

## 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/S1A2T00/ML--Pipeline.git
```

Move into the project directory:

```bash
cd ML--Pipeline
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## ☁️ AWS S3 Integration

AWS S3 is used as the DVC remote storage to securely store datasets and trained model artifacts. This enables:

* Dataset versioning
* Model versioning
* Easy collaboration
* Reproducible ML workflows

---

## 📊 Key Features

* Modular ML pipeline
* Configuration-driven architecture
* Data versioning with DVC
* Remote storage using AWS S3
* Reproducible workflow
* Logging and exception handling
* Git version control

---

## 📈 Future Improvements

* Model deployment using FastAPI
* Docker containerization
* CI/CD with GitHub Actions
* MLflow experiment tracking
* Kubernetes deployment

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Push the branch.
5. Open a Pull Request.

---

## 📬 Contact

**Satyam Pandey**

* LinkedIn: https://www.linkedin.com/in/satyam-pandey-cs
* GitHub: https://github.com/S1A2T00

---

⭐ If you found this project helpful, consider giving it a Star!
