from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import uvicorn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans, DBSCAN
import os
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(file: UploadFile = File(...), model: str = Form(...)):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, file.filename)

    with open(file_path, "wb") as f:
        f.write(await file.read())

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    target = df.columns[-1]
    X = df.drop(target, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

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

    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    return {
        "model_used": model,
        "target_column": target,
        "sample_predictions": predictions[:10].tolist()
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
