# Multi-Cancer Risk Assessment — Web Frontend

A Flask web app that puts a proper UI in front of the 10 Random Forest models from the notebooks project. Pick a cancer type from the left rail, fill in the intake chart, click **Run Assessment**, and get a Yes/No prediction with a confidence gauge.

## How to run

```
pip install -r requirements.txt
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## Structure

```
cancer_frontend/
├── app.py              # Flask backend — field schemas, preprocessing, /api/predict
├── templates/
│   └── index.html
├── static/
│   ├── css/style.css   # clinical-chart design system
│   └── js/main.js      # dynamic form rendering + gauge animation
├── models/              # trained .joblib models (copied from the notebooks project)
└── requirements.txt
```

## Design notes

The visual language is built around how clinical data is actually displayed — a dark navy index rail listing the 10 modules, an "intake chart" form styled like a lab report, and monospace (IBM Plex Mono) for every data field and readout to echo how vitals get logged. The result panel uses a semicircular gauge, needle-animated to the model's confidence, on a teal (low risk) → amber (high risk) arc.

Each module carries its data-quality tier (1/2/3) as a visible badge, matching the same honesty framing as the notebooks project — real clinical data, real-but-caveated data, or synthetic data, always disclosed rather than hidden.

## Note

This runs Flask's built-in dev server, which is fine for local use/demoing but says so in its own startup warning — don't deploy this as-is to the public internet. If you want it deployed (e.g. for a portfolio link), let me know and I can help set that up properly (gunicorn + a host like Render/Railway).
