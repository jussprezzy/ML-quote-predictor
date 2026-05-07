import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
import datetime
import warnings
warnings.filterwarnings('ignore')

# ── Dataset ────────────────────────────────────────────────────────────────────
DATA = pd.DataFrame([
    {"job_type": "Tar Asphalt Surfacing 30-40mm",   "area": "Kempton Park",  "client_type": "Commercial",  "quantity": 800,  "unit_price": 65, "quote_amount": 52000,  "outcome": "Won"},
    {"job_type": "Tar Asphalt Surfacing 30-40mm",   "area": "Tembisa",       "client_type": "Government",  "quantity": 1200, "unit_price": 58, "quote_amount": 69600,  "outcome": "Won"},
    {"job_type": "Parking Lot Resurface",            "area": "Midrand",       "client_type": "Commercial",  "quantity": 950,  "unit_price": 72, "quote_amount": 68400,  "outcome": "Won"},
    {"job_type": "Driveway Surfacing",               "area": "Edenvale",      "client_type": "Residential", "quantity": 200,  "unit_price": 78, "quote_amount": 15600,  "outcome": "Lost"},
    {"job_type": "Tar Asphalt Surfacing 30-40mm",   "area": "Boksburg",      "client_type": "Commercial",  "quantity": 1500, "unit_price": 62, "quote_amount": 93000,  "outcome": "Won"},
    {"job_type": "Road Section Resurface",           "area": "Tembisa",       "client_type": "Government",  "quantity": 2000, "unit_price": 55, "quote_amount": 110000, "outcome": "Won"},
    {"job_type": "Parking Lot Resurface",            "area": "Kempton Park",  "client_type": "Commercial",  "quantity": 700,  "unit_price": 75, "quote_amount": 52500,  "outcome": "Lost"},
    {"job_type": "Industrial Yard Surfacing",        "area": "Midrand",       "client_type": "Commercial",  "quantity": 1800, "unit_price": 68, "quote_amount": 122400, "outcome": "Won"},
    {"job_type": "Driveway Surfacing",               "area": "Kempton Park",  "client_type": "Residential", "quantity": 250,  "unit_price": 74, "quote_amount": 18500,  "outcome": "Won"},
    {"job_type": "Road Section Resurface",           "area": "Benoni",        "client_type": "Government",  "quantity": 2500, "unit_price": 53, "quote_amount": 132500, "outcome": "Lost"},
    {"job_type": "Tar Asphalt Surfacing 30-40mm",   "area": "Edenvale",      "client_type": "Commercial",  "quantity": 900,  "unit_price": 67, "quote_amount": 60300,  "outcome": "Won"},
    {"job_type": "Parking Lot Resurface",            "area": "Boksburg",      "client_type": "Commercial",  "quantity": 1100, "unit_price": 70, "quote_amount": 77000,  "outcome": "Won"},
    {"job_type": "Industrial Yard Surfacing",        "area": "Kempton Park",  "client_type": "Commercial",  "quantity": 2200, "unit_price": 71, "quote_amount": 156200, "outcome": "Won"},
    {"job_type": "Driveway Surfacing",               "area": "Midrand",       "client_type": "Residential", "quantity": 180,  "unit_price": 80, "quote_amount": 14400,  "outcome": "Lost"},
    {"job_type": "Road Section Resurface",           "area": "Tembisa",       "client_type": "Government",  "quantity": 3000, "unit_price": 52, "quote_amount": 156000, "outcome": "Won"},
    {"job_type": "Tar Asphalt Surfacing 30-40mm",   "area": "Kempton Park",  "client_type": "Commercial",  "quantity": 1000, "unit_price": 63, "quote_amount": 63000,  "outcome": "Won"},
    {"job_type": "Parking Lot Resurface",            "area": "Midrand",       "client_type": "Commercial",  "quantity": 850,  "unit_price": 73, "quote_amount": 62050,  "outcome": "Lost"},
    {"job_type": "Industrial Yard Surfacing",        "area": "Edenvale",      "client_type": "Commercial",  "quantity": 1600, "unit_price": 69, "quote_amount": 110400, "outcome": "Won"},
    {"job_type": "Driveway Surfacing",               "area": "Boksburg",      "client_type": "Residential", "quantity": 220,  "unit_price": 76, "quote_amount": 16720,  "outcome": "Won"},
    {"job_type": "Road Section Resurface",           "area": "Kempton Park",  "client_type": "Government",  "quantity": 1800, "unit_price": 56, "quote_amount": 100800, "outcome": "Won"},
    {"job_type": "Tar Asphalt Surfacing 30-40mm",   "area": "Benoni",        "client_type": "Commercial",  "quantity": 750,  "unit_price": 66, "quote_amount": 49500,  "outcome": "Won"},
    {"job_type": "Parking Lot Resurface",            "area": "Tembisa",       "client_type": "Government",  "quantity": 1300, "unit_price": 68, "quote_amount": 88400,  "outcome": "Won"},
    {"job_type": "Industrial Yard Surfacing",        "area": "Boksburg",      "client_type": "Commercial",  "quantity": 2000, "unit_price": 72, "quote_amount": 144000, "outcome": "Lost"},
    {"job_type": "Driveway Surfacing",               "area": "Edenvale",      "client_type": "Residential", "quantity": 300,  "unit_price": 75, "quote_amount": 22500,  "outcome": "Won"},
    {"job_type": "Road Section Resurface",           "area": "Midrand",       "client_type": "Government",  "quantity": 2200, "unit_price": 54, "quote_amount": 118800, "outcome": "Won"},
])

# ── Price ranges per job type ───────────────────────────────────────────────────
PRICE_RANGES = {
    "Tar Asphalt Surfacing 30-40mm":  {"min": 55, "sweet": 65, "max": 75},
    "Parking Lot Resurface":          {"min": 63, "sweet": 72, "max": 80},
    "Driveway Surfacing":             {"min": 68, "sweet": 76, "max": 84},
    "Road Section Resurface":         {"min": 48, "sweet": 55, "max": 62},
    "Industrial Yard Surfacing":      {"min": 62, "sweet": 70, "max": 78},
}

# ── Client return likelihood ───────────────────────────────────────────────────
CLIENT_RETURN = {
    "Government":  {"label": "High",   "days": 45,  "note": "Tenders re-open every 60–90 days — follow up early"},
    "Commercial":  {"label": "Medium", "days": 75,  "note": "Usually return within a quarter for next phase"},
    "Residential": {"label": "Low",    "days": 120, "note": "Seasonal work — follow up in 3–4 months"},
}

# ── Job forecast (next 30 days) ────────────────────────────────────────────────
def get_job_forecast():
    today = datetime.date.today()
    forecast = []
    job_types = list(PRICE_RANGES.keys())
    areas = ["Kempton Park", "Tembisa", "Midrand", "Edenvale", "Boksburg", "Benoni"]
    client_types = ["Commercial", "Government", "Residential"]
    np.random.seed(42)
    for i in range(8):
        days_ahead = np.random.randint(1, 31)
        forecast_date = today + datetime.timedelta(days=int(days_ahead))
        jt = job_types[i % len(job_types)]
        qty = np.random.randint(400, 2500)
        forecast.append({
            "date": forecast_date.strftime("%d %b %Y"),
            "job_type": jt,
            "area": areas[i % len(areas)],
            "client_type": client_types[i % 3],
            "estimated_value": f"R{PRICE_RANGES[jt]['sweet'] * qty:,.0f}",
            "confidence": f"{np.random.randint(65, 92)}%",
        })
    forecast.sort(key=lambda x: x["date"])
    return forecast

# ── Encoders & Model ────────────────────────────────────────────────────────────
le_job    = LabelEncoder()
le_area   = LabelEncoder()
le_client = LabelEncoder()

def prepare_features(df):
    df = df.copy()
    df["job_enc"]    = le_job.fit_transform(df["job_type"])
    df["area_enc"]   = le_area.fit_transform(df["area"])
    df["client_enc"] = le_client.fit_transform(df["client_type"])
    return df[["job_enc", "area_enc", "client_enc", "quantity", "unit_price", "quote_amount"]]

def train_model():
    df = DATA.copy()
    X = prepare_features(df)
    y = (df["outcome"] == "Won").astype(int)
    clf = GradientBoostingClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    return clf

MODEL = train_model()

# ── Public API ──────────────────────────────────────────────────────────────────
def predict_win_probability(job_type, area, client_type, quantity, unit_price):
    quote_amount = quantity * unit_price
    try:
        job_enc    = le_job.transform([job_type])[0]
        area_enc   = le_area.transform([area])[0]
        client_enc = le_client.transform([client_type])[0]
    except ValueError:
        job_enc = area_enc = client_enc = 0

    X = pd.DataFrame([{
        "job_enc": job_enc, "area_enc": area_enc, "client_enc": client_enc,
        "quantity": quantity, "unit_price": unit_price, "quote_amount": quote_amount,
    }])
    prob = MODEL.predict_proba(X)[0][1]
    pr = PRICE_RANGES.get(job_type, {"min": unit_price * 0.9, "sweet": unit_price, "max": unit_price * 1.1})
    cr = CLIENT_RETURN.get(client_type, CLIENT_RETURN["Commercial"])

    return {
        "win_probability":        round(prob * 100, 1),
        "quote_amount":           quote_amount,
        "recommended_min":        pr["min"],
        "recommended_sweet":      pr["sweet"],
        "recommended_max":        pr["max"],
        "client_return_likelihood": cr["label"],
        "client_return_days":     cr["days"],
        "client_return_note":     cr["note"],
    }

def get_all_options():
    return {
        "job_types":    sorted(DATA["job_type"].unique().tolist()),
        "areas":        sorted(DATA["area"].unique().tolist()),
        "client_types": sorted(DATA["client_type"].unique().tolist()),
    }

def get_historical_data():
    return DATA.copy()
