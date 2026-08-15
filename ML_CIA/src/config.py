"""Shared constants and variable metadata."""

from pathlib import Path

SEED = 42
TARGET = "Diabetes_binary"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "dataset" / "diabetes_binary_health_indicators_BRFSS2015.csv"
ARTIFACT_DIR = ROOT / "artifacts"
FIGURE_DIR = ROOT / "results" / "figures"
TABLE_DIR = ROOT / "results" / "tables"

BINARY_FEATURES = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
CONTINUOUS_FEATURES = ["BMI", "MentHlth", "PhysHlth"]
ORDINAL_FEATURES = ["GenHlth", "Age", "Education", "Income"]
RAW_FEATURES = BINARY_FEATURES + CONTINUOUS_FEATURES + ORDINAL_FEATURES
ENGINEERED_NUMERIC = [
    "HealthBurden", "CardioBurden", "LifestyleScore", "AccessBarrier", "BMIAgeRisk"
]
ENGINEERED_CATEGORICAL = ["BMIGroup"]

VALID_RANGES = {
    **{column: (0, 1) for column in BINARY_FEATURES},
    "BMI": (10, 100),
    "MentHlth": (0, 30),
    "PhysHlth": (0, 30),
    "GenHlth": (1, 5),
    "Age": (1, 13),
    "Education": (1, 6),
    "Income": (1, 8),
}

LABELS = {
    "HighBP": "High blood pressure",
    "HighChol": "High cholesterol",
    "CholCheck": "Cholesterol checked in past 5 years",
    "BMI": "Body mass index (BMI)",
    "Smoker": "Smoked at least 100 cigarettes",
    "Stroke": "History of stroke",
    "HeartDiseaseorAttack": "Heart disease or prior heart attack",
    "PhysActivity": "Physical activity in past 30 days",
    "Fruits": "Fruit at least once daily",
    "Veggies": "Vegetables at least once daily",
    "HvyAlcoholConsump": "Heavy alcohol consumption",
    "AnyHealthcare": "Has healthcare coverage",
    "NoDocbcCost": "Could not see doctor because of cost",
    "GenHlth": "General health (1 excellent–5 poor)",
    "MentHlth": "Poor mental-health days (past 30)",
    "PhysHlth": "Poor physical-health days (past 30)",
    "DiffWalk": "Serious difficulty walking/climbing stairs",
    "Sex": "Sex recorded by BRFSS (0 female, 1 male)",
    "Age": "Age category (1 youngest–13 oldest)",
    "Education": "Education category (1 lowest–6 highest)",
    "Income": "Income category (1 lowest–8 highest)",
}

