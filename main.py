from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, confusion_matrix, silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import io
import os
import tempfile

# Dark theme
plt.style.use("dark_background")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Convert Matplotlib figure to Base64
def fig_to_base64():
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    # Load data
    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    target = df.columns[-1]
    X = df.drop(target, axis=1)
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # Select model
    if model == "linear_regression":
        clf = LinearRegression()

    elif model == "random_forest_regressor":
        clf = RandomForestRegressor()

    elif model == "logistic_regression":
        clf = LogisticRegression(max_iter=500)

    elif model == "random_forest_classifier":
        clf = RandomForestClassifier()

    elif model == "kmeans":
        clf = KMeans(n_clusters=3)

    elif model == "dbscan":
        clf = DBSCAN()

    elif model == "automl":
        models = [
            ("Linear Regression", LinearRegression()),
            ("Random Forest Regressor", RandomForestRegressor()),
            ("Random Forest Classifier", RandomForestClassifier())
        ]
        best_score = -999
        best_model = None

        for name, m in models:
            try:
                m.fit(X_train, y_train)
                score = m.score(X_test, y_test)
                if score > best_score:
                    best_score = score
                    best_model = m
            except:
                pass

        clf = best_model

    else:
        return {"error": "Unknown model"}

    # Fit model
    clf.fit(X_train, y_train)

    # Predictions
    predictions = clf.predict(X_test)

    # Metrics
    metrics = {
        "r2": None,
        "mae": None,
        "mse": None,
        "accuracy": None,
        "silhouette": None
    }

    # Regression metrics
    try:
        metrics["r2"] = r2_score(y_test, predictions)
        metrics["mae"] = mean_absolute_error(y_test, predictions)
        metrics["mse"] = mean_squared_error(y_test, predictions)
    except:
        pass

    # Classification metrics
    try:
        metrics["accuracy"] = accuracy_score(y_test, predictions.round())
    except:
        pass

    # Clustering metrics
    try:
        metrics["silhouette"] = silhouette_score(X, clf.labels_)
    except:
        pass

    # Generate charts
    charts = {
        "actual_vs_predicted": None,
        "residuals": None,
        "feature_importance": None,
        "confusion_matrix": None,
        "clusters": None
    }

    # Actual vs Predicted
    try:
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, color="cyan")
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Actual vs Predicted")
        charts["actual_vs_predicted"] = fig_to_base64()
        plt.close()
    except:
        pass

    # Residual plot
    try:
        residuals = y_test - predictions
        plt.figure(figsize=(8, 6))
        plt.scatter(predictions, residuals, color="orange")
        plt.axhline(0, color="white", linestyle="--")
        plt.xlabel("Predicted")
        plt.ylabel("Residuals")
        plt.title("Residual Plot")
        charts["residuals"] = fig_to_base64()
        plt.close()
    except:
        pass

    # Feature importance
    try:
        if hasattr(clf, "feature_importances_"):
            plt.figure(figsize=(8, 6))
            sns.barplot(x=clf.feature_importances_, y=X.columns, palette="viridis")
            plt.title("Feature Importance")
            charts["feature_importance"] = fig_to_base64()
            plt.close()
    except:
        pass

    # Confusion matrix
    try:
        cm = confusion_matrix(y_test, predictions.round())
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, cmap="magma", fmt="d")
        plt.title("Confusion Matrix")
        charts["confusion_matrix"] = fig_to_base64()
        plt.close()
    except:
        pass

    # Clusters
    try:
        if hasattr(clf, "labels_"):
            plt.figure(figsize=(8, 6))
            plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=clf.labels_, cmap="cool")
            plt.title("Cluster Plot")
            charts["clusters"] = fig_to_base64()
            plt.close()
    except:
        pass

    return {
        "model_used": model,
        "target_column": target,
        "predictions": predictions[:20].tolist(),
        "metrics": metrics,
        "charts": charts
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
