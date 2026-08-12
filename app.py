"""
Multi-Cancer Risk Assessment — Flask backend.
Serves the frontend and exposes /api/predict, loading the RF models
trained in the companion notebooks project.
"""
from flask import Flask, render_template, request, jsonify
import joblib
import os
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = Flask(__name__)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# ---------------------------------------------------------------------------
# Field schemas — mirrors the cleaning/encoding logic in each notebook.
# type: "number" | "select"
# ---------------------------------------------------------------------------

CANCERS = {
    "breast": {
        "label": "Breast Cancer",
        "tier": 1,
        "tier_note": "Real clinical data (UCI Wisconsin Diagnostic, 569 patients)",
        "model_file": "breast_cancer_rf.joblib",
        "fields": [
            {"name": f, "label": f, "type": "number", "step": "any"} for f in [
                "mean radius","mean texture","mean perimeter","mean area","mean smoothness",
                "mean compactness","mean concavity","mean concave points","mean symmetry","mean fractal dimension",
                "radius error","texture error","perimeter error","area error","smoothness error",
                "compactness error","concavity error","concave points error","symmetry error","fractal dimension error",
                "worst radius","worst texture","worst perimeter","worst area","worst smoothness",
                "worst compactness","worst concavity","worst concave points","worst symmetry","worst fractal dimension"
            ]
        ],
    },
    "cervical": {
        "label": "Cervical Cancer",
        "tier": 1,
        "tier_note": "Real clinical data (UCI Risk Factors, 858 patients)",
        "model_file": "cervical_cancer_rf.joblib",
        "fields": [
            {"name": f, "label": f, "type": "number", "step": "any"} for f in [
                "Age","Number of sexual partners","First sexual intercourse","Num of pregnancies",
                "Smokes","Smokes (years)","Smokes (packs/year)","Hormonal Contraceptives",
                "Hormonal Contraceptives (years)","IUD","IUD (years)","STDs","STDs (number)"
            ]
        ],
    },
    "thyroid": {
        "label": "Thyroid Cancer (Recurrence)",
        "tier": 1,
        "tier_note": "Real clinical data (UCI DTC Recurrence, 383 patients)",
        "model_file": "thyroid_cancer_rf.joblib",
        "encoders_file": "thyroid_cancer_encoders.joblib",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
            {"name": "Gender", "label": "Gender", "type": "select", "options": ["F", "M"]},
            {"name": "Smoking", "label": "Smoking", "type": "select", "options": ["No", "Yes"]},
            {"name": "Hx Smoking", "label": "History of Smoking", "type": "select", "options": ["No", "Yes"]},
            {"name": "Hx Radiothreapy", "label": "History of Radiotherapy", "type": "select", "options": ["No", "Yes"]},
            {"name": "Thyroid Function", "label": "Thyroid Function", "type": "select",
             "options": ["Euthyroid", "Clinical Hyperthyroidism", "Clinical Hypothyroidism", "Subclinical Hyperthyroidism", "Subclinical Hypothyroidism"]},
            {"name": "Physical Examination", "label": "Physical Examination", "type": "select",
             "options": ["Single nodular goiter-left", "Single nodular goiter-right", "Multinodular goiter", "Normal", "Diffuse goiter"]},
            {"name": "Adenopathy", "label": "Adenopathy", "type": "select", "options": ["No", "Right", "Left", "Bilateral", "Posterior", "Extensive"]},
            {"name": "Pathology", "label": "Pathology", "type": "select", "options": ["Micropapillary", "Papillary", "Follicular", "Hurthel cell"]},
            {"name": "Focality", "label": "Focality", "type": "select", "options": ["Uni-Focal", "Multi-Focal"]},
            {"name": "Risk", "label": "Risk", "type": "select", "options": ["Low", "Intermediate", "High"]},
            {"name": "T", "label": "Tumor stage (T)", "type": "select", "options": ["T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"]},
            {"name": "N", "label": "Node stage (N)", "type": "select", "options": ["N0", "N1a", "N1b"]},
            {"name": "M", "label": "Metastasis stage (M)", "type": "select", "options": ["M0", "M1"]},
            {"name": "Stage", "label": "Overall Stage", "type": "select", "options": ["I", "II", "III", "IVA", "IVB"]},
            {"name": "Response", "label": "Response to Treatment", "type": "select",
             "options": ["Excellent", "Indeterminate", "Biochemical Incomplete", "Structural Incomplete"]},
        ],
    },
    "liver": {
        "label": "Liver Disease",
        "tier": 2,
        "tier_note": "Real data (UCI ILPD) — liver disease broadly, not cancer-specific",
        "model_file": "liver_disease_rf.joblib",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
            {"name": "Gender", "label": "Gender", "type": "select", "options": ["Male", "Female"]},
            {"name": "Total_Bilirubin", "label": "Total Bilirubin", "type": "number", "step": "any"},
            {"name": "Direct_Bilirubin", "label": "Direct Bilirubin", "type": "number", "step": "any"},
            {"name": "Alkaline_Phosphotase", "label": "Alkaline Phosphotase", "type": "number", "step": "any"},
            {"name": "Alamine_Aminotransferase", "label": "Alamine Aminotransferase (SGPT)", "type": "number", "step": "any"},
            {"name": "Aspartate_Aminotransferase", "label": "Aspartate Aminotransferase (SGOT)", "type": "number", "step": "any"},
            {"name": "Total_Protiens", "label": "Total Proteins", "type": "number", "step": "any"},
            {"name": "Albumin", "label": "Albumin", "type": "number", "step": "any"},
            {"name": "Albumin_and_Globulin_Ratio", "label": "Albumin/Globulin Ratio", "type": "number", "step": "any"},
        ],
    },
    "prostate": {
        "label": "Prostate Cancer",
        "tier": 2,
        "tier_note": "Real data, small sample (100 patients)",
        "model_file": "prostate_cancer_rf.joblib",
        "fields": [
            {"name": f, "label": f.replace("_", " ").title(), "type": "number", "step": "any"}
            for f in ["radius","texture","perimeter","area","smoothness","compactness","symmetry","fractal_dimension"]
        ],
    },
    "lung": {
        "label": "Lung Cancer",
        "tier": 2,
        "tier_note": "Real survey data — symptom self-report, not lab values",
        "model_file": "lung_cancer_rf.joblib",
        "fields": [
            {"name": "GENDER", "label": "Gender", "type": "select", "options": ["Male", "Female"]},
            {"name": "AGE", "label": "Age", "type": "number", "step": "1"},
        ] + [
            {"name": f, "label": f.strip().replace("_", " ").title(), "type": "select", "options": ["No", "Yes"]}
            for f in ["SMOKING","YELLOW_FINGERS","ANXIETY","PEER_PRESSURE","CHRONIC DISEASE","FATIGUE",
                      "ALLERGY","WHEEZING","ALCOHOL CONSUMING","COUGHING","SHORTNESS OF BREATH",
                      "SWALLOWING DIFFICULTY","CHEST PAIN"]
        ],
    },
    "oral": {
        "label": "Oral Cancer",
        "tier": 3,
        "tier_note": "Synthetic data — no real public dataset exists for this task",
        "model_file": "oral_cancer_rf.joblib",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
            {"name": "Gender", "label": "Gender", "type": "select", "options": ["Male", "Female"]},
        ] + [
            {"name": f, "label": f.replace("_", " "), "type": "select", "options": ["No", "Yes"]}
            for f in ["Tobacco_Use","Alcohol_Consumption","HPV_Infection","Betel_Quid_Use","Poor_Oral_Hygiene",
                      "Family_History_Cancer","Oral_Lesions","White_Red_Patches","Difficulty_Swallowing"]
        ],
    },
    "colorectal": {
        "label": "Colorectal Cancer",
        "tier": 3,
        "tier_note": "Synthetic data — public versions of this dataset are also undisclosed synthetic data",
        "model_file": "colorectal_cancer_rf.joblib",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
        ] + [
            {"name": f, "label": f.replace("_", " "), "type": "select", "options": ["No", "Yes"]}
            for f in ["Family_History","Red_Meat_Diet","Low_Fiber_Diet","Obesity","Smoking","Alcohol_Use",
                      "Sedentary_Lifestyle","IBD_History","Blood_In_Stool","Polyps_History"]
        ],
    },
    "skin": {
        "label": "Skin Cancer",
        "tier": 3,
        "tier_note": "Synthetic data — real skin cancer screening is image-based, not tabular",
        "model_file": "skin_cancer_rf.joblib",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
        ] + [
            {"name": f, "label": f.replace("_", " "), "type": "select", "options": ["No", "Yes"]}
            for f in ["Fair_Skin","High_Sun_Exposure","Sunburn_History","Family_History","Many_Moles",
                      "Irregular_Border","Asymmetry","Color_Variation","Diameter_Over_6mm","Evolving_Lesion"]
        ],
    },
    "kidney": {
        "label": "Kidney Cancer",
        "tier": 3,
        "tier_note": "Synthetic data — real kidney cancer data is CT-image/genomic, not tabular",
        "model_file": "kidney_cancer_rf.joblib",
        "fields": [
            {"name": "Age", "label": "Age", "type": "number", "step": "1"},
        ] + [
            {"name": f, "label": f.replace("_", " "), "type": "select", "options": ["No", "Yes"]}
            for f in ["Smoking","Obesity","Hypertension","Family_History","Dialysis_History",
                      "Blood_In_Urine","Flank_Pain","Unexplained_Weight_Loss","Palpable_Mass"]
        ],
    },
}

# Fix the FATIGUE / ALLERGY trailing-space field names to match training columns exactly
_LUNG_RENAME = {"FATIGUE": "FATIGUE ", "ALLERGY": "ALLERGY "}
for f in CANCERS["lung"]["fields"]:
    if f["name"] in _LUNG_RENAME:
        f["name"] = _LUNG_RENAME[f["name"]]

_model_cache = {}
_encoder_cache = {}


def get_model(key):
    if key not in _model_cache:
        _model_cache[key] = joblib.load(os.path.join(MODELS_DIR, CANCERS[key]["model_file"]))
    return _model_cache[key]


def get_encoders(key):
    if key not in _encoder_cache:
        enc_file = CANCERS[key].get("encoders_file")
        _encoder_cache[key] = joblib.load(os.path.join(MODELS_DIR, enc_file)) if enc_file else None
    return _encoder_cache[key]


def build_feature_vector(key, form_values):
    """Replicates each notebook's preprocessing to build the row the model expects."""
    cfg = CANCERS[key]
    encoders = get_encoders(key)
    row = []
    for field in cfg["fields"]:
        name = field["name"]
        raw = form_values.get(name, "")
        if field["type"] == "number":
            val = float(raw)
        else:  # select
            if key == "thyroid" and name != "Age":
                le = encoders[name]
                val = le.transform([raw])[0] if raw in le.classes_ else 0
            elif name in ("Gender", "GENDER"):
                val = 1 if raw == "Male" else 0
            else:
                val = 1 if raw == "Yes" else 0
        row.append(val)
    return row


@app.route("/")
def index():
    return render_template("index.html", cancers=CANCERS)


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()
    key = data.get("cancer")
    if key not in CANCERS:
        return jsonify({"error": "Unknown cancer type"}), 400

    try:
        row = build_feature_vector(key, data.get("values", {}))
    except Exception as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    model = get_model(key)
    pred = int(model.predict([row])[0])
    prob = float(model.predict_proba([row])[0][1])

    return jsonify({
        "prediction": "Yes" if pred == 1 else "No",
        "confidence": round(prob * 100, 1),
        "tier": CANCERS[key]["tier"],
        "tier_note": CANCERS[key]["tier_note"],
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
