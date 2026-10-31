import logging
import sys
import time
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from pydantic import BaseModel

# Step 4: Configure the Application Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

MODEL_FILE = "student_result_model.pkl"

# Step 5: Log Model Loading
try:
    model = joblib.load(MODEL_FILE)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise

app = FastAPI(title="Student Result Prediction API")


class StudentResultInput(BaseModel):
    attendance: float
    internal_marks: float
    assignment_marks: float
    previous_score: float


# Step 6: Add Request/Response Logging Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    try:
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        level = logging.WARNING if response.status_code >= 400 else logging.INFO

        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            request.method,
            request.url.path,
            duration_ms,
        )
        raise


# Step 7: Log FastAPI Validation Failures
@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    fields = [
        ".".join(str(part) for part in error.get("loc", []))
        for error in exc.errors()
    ]
    logger.warning(
        "event=validation_error path=%s field_locations=%s error_count=%s",
        request.url.path,
        fields,
        len(exc.errors()),
    )
    return await request_validation_exception_handler(request, exc)


# Step 9: Health Endpoint
@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "student-result-prediction"
    }


# Step 8: Add Safe Prediction Logging
SAFE_INPUT_FIELDS = set()


@app.post("/predict")
def predict(data: StudentResultInput):
    try:
        payload = data.model_dump() if hasattr(data, "model_dump") else data.dict()
        safe_input = {
            key: value
            for key, value in payload.items()
            if key in SAFE_INPUT_FIELDS
        }

        logger.info(
            "event=prediction_request feature_names=%s safe_input=%s",
            sorted(payload.keys()),
            safe_input,
        )

        features_order = [
            "attendance",
            "internal_marks",
            "assignment_marks",
            "previous_score",
        ]
        input_df = pd.DataFrame([{k: payload[k] for k in features_order}])
        prediction_code = int(model.predict(input_df)[0])
        prediction_label = "PASS" if prediction_code == 1 else "FAIL"

        logger.info(
            "event=prediction_success prediction=%s",
            prediction_label,
        )
        return {
            "prediction": prediction_label,
            "prediction_code": prediction_code,
        }

    except Exception:
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
