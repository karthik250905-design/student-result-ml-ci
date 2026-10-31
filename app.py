import logging
import sys
import time
import uuid
import joblib
import pandas as pd
from fastapi import FastAPI, Request, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.responses import Response
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Practical-11 Logger Configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("student-result-api")

# Step 5: Define Application Metrics
HTTP_REQUESTS = Counter(
    "student_api_http_requests_total",
    "Total HTTP requests received by the Student Result Prediction API",
    ["method", "path", "status"]
)

HTTP_LATENCY = Histogram(
    "student_api_http_request_duration_seconds",
    "HTTP request duration in seconds",
    ["method", "path"]
)

PREDICTIONS = Counter(
    "student_api_predictions_total",
    "Total Student Result Prediction attempts",
    ["result"]
)

MODEL_LOADED = Gauge(
    "student_api_model_loaded",
    "Whether the Student Result Prediction model is loaded"
)

MODEL_FILE = "student_result_model.pkl"

# Step 6: Connect Model-Loaded Gauge to Existing Model Loading
try:
    model = joblib.load(MODEL_FILE)
    MODEL_LOADED.set(1)
    logger.info("event=model_loaded model_file=%s", MODEL_FILE)
except Exception:
    MODEL_LOADED.set(0)
    logger.exception("event=model_load_failed model_file=%s", MODEL_FILE)
    raise

app = FastAPI(title="Student Result Prediction API")


class StudentResultInput(BaseModel):
    attendance: float
    internal_marks: float
    assignment_marks: float
    previous_score: float


# Step 7: Add Metrics to Existing Request Middleware
@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    path = request.url.path
    method = request.method
    start_time = time.perf_counter()

    try:
        response = await call_next(request)
        duration = time.perf_counter() - start_time
        duration_ms = duration * 1000

        # Do not count Prometheus' own scrape requests in user-traffic metrics
        if path != "/metrics":
            HTTP_REQUESTS.labels(
                method=method,
                path=path,
                status=str(response.status_code)
            ).inc()
            HTTP_LATENCY.labels(
                method=method,
                path=path
            ).observe(duration)

        level = logging.WARNING if response.status_code >= 400 else logging.INFO
        logger.log(
            level,
            "event=http_request request_id=%s method=%s path=%s status=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            response.status_code,
            duration_ms,
        )
        response.headers["X-Request-ID"] = request_id
        return response

    except Exception:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.exception(
            "event=http_request_failed request_id=%s method=%s path=%s duration_ms=%.2f",
            request_id,
            method,
            path,
            duration_ms,
        )
        raise


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


@app.get("/")
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "student-result-prediction"
    }


# Step 9: Add the /metrics Endpoint
@app.get("/metrics")
def metrics():
    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


SAFE_INPUT_FIELDS = set()


# Step 8: Count Prediction Successes and Runtime Failures
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

        PREDICTIONS.labels(result="success").inc()
        logger.info("event=prediction_succeeded prediction=%s", prediction_label)

        return {
            "prediction": prediction_label,
            "prediction_code": prediction_code,
        }

    except Exception:
        PREDICTIONS.labels(result="error").inc()
        logger.exception("event=prediction_failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
