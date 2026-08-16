from pydantic import BaseModel, ConfigDict, Field


class ClientPredictionRequest(BaseModel):
    """
    One client's attributes, known BEFORE the call is made.
    Deliberately does NOT include `duration`
    """
 
    age: int = Field(..., ge=18, le=100, description="Client age in years")
    job: str = Field(..., description="Job category, e.g. 'management', 'technician'")
    marital: str = Field(..., description="'married', 'single', or 'divorced'")
    education: str = Field(..., description="'primary', 'secondary', 'tertiary', or 'unknown'")
    default: str = Field(..., description="'yes'/'no' - has credit in default")
    balance: float = Field(..., description="Average yearly account balance")
    housing: str = Field(..., description="'yes'/'no' - has a housing loan")
    loan: str = Field(..., description="'yes'/'no' - has a personal loan")
    contact: str = Field(..., description="Contact channel: 'cellular', 'telephone', 'unknown'")
    day: int = Field(..., ge=1, le=31, description="Day of month of last contact")
    month: str = Field(..., description="Month of last contact, lowercase 3-letter e.g. 'may'")
    campaign: int = Field(..., ge=1, description="Number of contacts in the current campaign")
    pdays: int = Field(..., description="Days since last previous contact, or -1 if never contacted")
    previous: int = Field(..., ge=0, description="Number of contacts before this campaign")
    poutcome: str = Field(..., description="Outcome of previous campaign: 'success','failure','other','unknown'")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "age": 42, "job": "technician", "marital": "married", "education": "secondary",
            "default": "no", "balance": 1500.0, "housing": "yes", "loan": "no",
            "contact": "cellular", "day": 15, "month": "may", "campaign": 2,
            "pdays": -1, "previous": 0, "poutcome": "unknown",
        }
    })


class ClientPredictionResponse(BaseModel):
    subscription_probability: float = Field(..., description="Model's predicted probability of subscribing")
    decision_threshold_used: float = Field(..., description="The tuned threshold applied to reach the label")
    predicted_label: str = Field(..., description="'yes' or 'no', derived from probability vs threshold")   