## Wine Quality Prediction – End-to-End Machine Learning Pipeline

End-to-end machine learning pipeline that predicts wine quality based on physicochemical properties using a fully modular, production-ready ML workflow.

### Overview

This project builds a complete ML pipeline that predicts wine quality scores using chemical attributes such as acidity, sugar content, sulphates, alcohol percentage, and more.
The system is designed following scalable MLOps practices, including configuration-driven architecture, modular components, reproducible experiments, and MLflow tracking.

### Dataset

The model is trained on the popular Wine Quality dataset, containing the following features: fixed acidity, volatile acidity, citric acid, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide, density, pH,  sulphates, alcohol, Target: quality (0–10 scale)

### Features
- Modular & reusable ML pipeline components
- Configuration-based design (YAML-driven)
- Automated:
    1. Data Ingestion
    2. Data Validation
    3. Data Transformation
    4. Model Trainer
    5. Model Evaluation
- Experiment tracking with MLflow

### Technologies Used
- **Python**: Programming language used for developing the model and web application.
- **Flask**: Framework used for creating the web application.
- **Scikit-learn**: Machine learning library used for building and evaluating the models.
- **Pandas & Numpy**: Libraries for data manipulation and analysis.
- **Matplotlib & Seaborn**: Libraries for data visualization.
- **MLflow**: Experiment tracking
- **Dagshub**: Manage ML projects (Remote Repository to manage everything from data, code)