import os
import time
import psycopg2
import pandas as pd
from flask import Flask, request, jsonify, abort
from sklearn.utils import Bunch
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from typing import cast


class IrisModel:
    def __init__(self):
        self.iris: Bunch = cast(Bunch, load_iris(as_frame=True))

        X: pd.DataFrame = self.iris.data
        y: pd.Series = self.iris.target

        self.model = KNeighborsClassifier(n_neighbors=3)
        print("Loading the iris model...")
        start_time = time.time()

        self.model.fit(X, y)

        elapsed = time.time() - start_time
        print(f"Model loaded in {elapsed:.4f} seconds")

        self.feature_names: list[str] = list(self.iris.feature_names)
        self.target_names: list[str] = list(self.iris.target_names)


api = Flask(__name__)
hosted_model = IrisModel()


DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
    )

# Ensure predictions table exists
with get_db_connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS iris_predictions (
                id SERIAL PRIMARY KEY,
                sepal_length_cm FLOAT NOT NULL,
                sepal_width_cm FLOAT NOT NULL,
                petal_length_cm FLOAT NOT NULL,
                petal_width_cm FLOAT NOT NULL,
                predicted_class VARCHAR(50) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
    conn.commit()


@api.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a single JSON object with Iris features
    and returns the predicted class.
    Also inserts the prediction into db.
    """

    # Validate input JSON
    data = request.get_json()
    if not data:
        abort(400, "Request must contain JSON")

    # Make sure all required features are present
    missing = [f for f in hosted_model.feature_names if f not in data]
    if missing:
        abort(400, f"Missing required features: {missing}")

    # Prepare input for model
    X = pd.DataFrame([data], columns=hosted_model.feature_names)

    # Run inference
    pred_idx = hosted_model.model.predict(X)[0]
    pred_class = hosted_model.target_names[pred_idx]

    # Insert into Postgres
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO iris_predictions 
                    (sepal_length_cm, sepal_width_cm, petal_length_cm, petal_width_cm, predicted_class) 
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id;
                    """, 
                    (
                        data["sepal length (cm)"],
                        data["sepal width (cm)"],
                        data["petal length (cm)"],
                        data["petal width (cm)"],
                        pred_class,
                    )
                )
                row_id = cur.fetchone()[0] # type: ignore
            conn.commit()
    except Exception as e:
        abort(500, f"Database error: {str(e)}")

    # Respond with JSON
    return jsonify({
        "message": f"Prediction inserted into DB with id {row_id}",
        "prediction": pred_class
    })


if __name__ == "__main__":
    api.run(
        debug=os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on"),
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000"))
    )
