# app.py
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
import os

app = Flask(__name__)


class ModelFactory:
    @staticmethod
    def get_models():
        return {
            "knn": ("K-Nearest Neighbors", KNeighborsClassifier()),
            "naive_bayes": ("Naive Bayes", GaussianNB()),
            "decision_tree": ("Decision Tree", DecisionTreeClassifier(random_state=42)),
            "svm": ("Support Vector Machine", SVC(random_state=42)),
            "random_forest": ("Random Forest", RandomForestClassifier(random_state=42)),
            "logistic_regression": (
                "Logistic Regression",
                LogisticRegression(random_state=42),
            ),
            "gaussian_mixture": (
                "Gaussian Mixture",
                Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("gmm", GaussianMixture(random_state=42)),
                    ]
                ),
            ),
        }


def create_plot(train_sizes, accuracies, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes * 100, accuracies, marker="o")
    plt.xlabel("Training Data Size (%)")
    plt.ylabel("Model Accuracy")
    plt.title(f"{model_name} Accuracy vs Training Data Size")
    plt.grid(True)

    img = io.BytesIO()
    plt.savefig(img, format="png", bbox_inches="tight")
    plt.close()
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"})

    try:
        df = pd.read_csv(file)
        columns = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            columns.append(
                {
                    "name": col,
                    "dtype": dtype,
                    "is_numeric": dtype in ["int64", "float64"],
                }
            )
        return jsonify({"columns": columns})
    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        # Get data from request
        file = request.files["file"]
        features = request.form.getlist("features[]")
        label = request.form.get("label")
        encoding_methods = request.form.getlist("encoding_methods[]")
        encode_columns = request.form.getlist("encode_columns[]")
        min_split = float(request.form.get("min_split", 0.1))
        max_split = float(request.form.get("max_split", 0.9))
        split_step = float(request.form.get("split_step", 0.1))

        # Read CSV
        df = pd.read_csv(file)
        X = df[features]
        y = df[label]

        # Prepare encoders
        encoders = []
        categorical_features = []
        numeric_features = []

        for feature in features:
            if feature in encode_columns:
                categorical_features.append(feature)
            else:
                numeric_features.append(feature)

        # Create preprocessing pipeline based on encoding methods
        preprocessors = []
        if numeric_features:
            preprocessors.append(("num", StandardScaler(), numeric_features))

        if categorical_features:
            for method in encoding_methods:
                if method == "label":
                    preprocessors.append(
                        ("cat_label", LabelEncoder(), categorical_features)
                    )
                elif method == "onehot":
                    preprocessors.append(
                        (
                            "cat_onehot",
                            OneHotEncoder(sparse=False, handle_unknown="ignore"),
                            categorical_features,
                        )
                    )

        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=preprocessors, remainder="passthrough"
        )

        # Prepare training splits
        train_sizes = np.arange(min_split, max_split + split_step, split_step)

        # Initialize models
        models = ModelFactory.get_models()
        results = {}

        # Train and evaluate each model
        for model_key, (model_name, model) in models.items():
            accuracies = []

            for size in train_sizes:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, train_size=size, random_state=42
                )

                # Preprocess data
                X_train_processed = preprocessor.fit_transform(X_train)
                X_test_processed = preprocessor.transform(X_test)

                # Train and evaluate
                model.fit(X_train_processed, y_train)
                if isinstance(model, Pipeline) and isinstance(
                    model.named_steps["gmm"], GaussianMixture
                ):
                    y_pred = model.predict(X_test_processed)
                else:
                    y_pred = model.predict(X_test_processed)
                accuracy = accuracy_score(y_test, y_pred)
                accuracies.append(accuracy)

            # Create plot for this model
            plot = create_plot(train_sizes, accuracies, model_name)

            results[model_key] = {
                "name": model_name,
                "accuracies": accuracies,
                "plot": plot,
            }

        return jsonify({"results": results, "train_sizes": train_sizes.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
