from pathlib import Path
import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Student Result Prediction API",
    description="MLOps Student Result Prediction Service deployed from GHCR to Render",
    version="1.0.0"
)

MODEL_PATH = Path("student_result_model.pkl")

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


class StudentInput(BaseModel):
    attendance: float = Field(90, example=90)
    internal_marks: float = Field(85, example=85)
    assignment_marks: float = Field(88, example=88)
    previous_score: float = Field(80, example=80)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )
    return joblib.load(MODEL_PATH)


@app.get("/")
def root():
    return {
        "service": "student-result-prediction",
        "status": "running",
        "version": "1.0.0",
        "message": "Student Result Prediction API is live on Render"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "student-result-prediction"
    }


@app.post("/predict")
def predict(payload: StudentInput):
    sample = pd.DataFrame([{
        "attendance": payload.attendance,
        "internal_marks": payload.internal_marks,
        "assignment_marks": payload.assignment_marks,
        "previous_score": payload.previous_score
    }])

    model = load_model()
    prediction_code = int(model.predict(sample)[0])
    prediction = "PASS" if prediction_code == 1 else "FAIL"

    return {
        "prediction": prediction,
        "prediction_code": prediction_code
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
