import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import load_model, predict_employee_attrition

# Setup basic logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="HR Retention ML API",
    description="API for predicting employee attrition using a trained ML model.",
    version="1.0.0"
)

# Load the model and encoder at startup
model, explainer, encoder = load_model("models/model.pkl", "models/encoder.pkl")

# Pydantic Model strictly constrains allowed values to prevent "Garbage In, Garbage Out".
class EmployeeData(BaseModel):
    satisfaction_level: float = Field(..., ge=0.0, le=1.0, description="Satisfaction level between 0 and 1")
    last_evaluation: float = Field(..., ge=0.0, le=1.0, description="Last evaluation score between 0 and 1")
    number_project: int = Field(..., gt=0, le=20, description="Number of projects")
    average_monthly_hours: int = Field(..., gt=0, le=800, description="Average monthly hours worked")
    tenure: int = Field(..., gt=0, le=50, description="Years spent at the company")
    work_accident: int = Field(..., ge=0, le=1, description="0 or 1 if experienced a work accident")
    promotion_last_5years: int = Field(..., ge=0, le=1, description="0 or 1 if promoted in the last 5 years")
    department: str = Field(..., description="Department name (e.g., sales, IT, HR)")
    salary: str = Field(..., description="Salary level (low, medium, high)")

@app.post("/predict")
def predict_attrition(employee: EmployeeData):
    """
    Predict attrition probability for a given employee and
    return top driving factors using SHAP values.
    # Note: Removed 'async def' to avoid blocking issues during CPU-intensive SHAP/Prediction steps.
    """
    if model is None or encoder is None:
        logger.error("Failed to execute prediction - ML assets (model/encoder) are missing.")
        raise HTTPException(status_code=500, detail="Server misconfiguration: Models are currently unavailable.")

    try:
        emp_dict = employee.model_dump() # Pydantic v2 compliant (used to be .dict())
        result = predict_employee_attrition(model, explainer, encoder, emp_dict)
        return result
    except Exception as e:
        # Secure Error Handling: Log internally, generic error externally.
        logger.error(f"Prediction system error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal processing error occurred.")

@app.get("/")
def read_root():
    return {"message": "Welcome to the HR Retention API. Use POST /predict to get attrition predictions."}
