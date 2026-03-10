"""
main.py
-------
FastAPI application for the World Energy Consumption ML project.
Serves HTML pages via Jinja2 templates and exposes prediction endpoints.

Run:
    uvicorn app.main:app --reload
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── Setup ─────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')

app = FastAPI(title="Energy Consumption Predictor")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
templates.env.filters['enumerate'] = enumerate

# ── Load artefacts at startup ─────────────────────────────────────────────────
def load_artefacts():
    models = {}
    model_names = ['ridge', 'svr', 'randomforest', 'gradientboosting', 'gpr', 'mlp']
    for name in model_names:
        path = os.path.join(MODELS_DIR, f'{name}.pkl')
        if os.path.exists(path):
            models[name] = joblib.load(path)

    results, meta, feature_importance = {}, {}, {}

    results_path = os.path.join(MODELS_DIR, 'results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

    meta_path = os.path.join(MODELS_DIR, 'meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    fi_path = os.path.join(MODELS_DIR, 'feature_importance.json')
    if os.path.exists(fi_path):
        with open(fi_path) as f:
            feature_importance = json.load(f)

    return models, results, meta, feature_importance


MODELS, RESULTS, META, FEATURE_IMPORTANCE = load_artefacts()

MODEL_DISPLAY = {
    'ridge': 'Ridge Regression',
    'svr': 'Support Vector Regression',
    'randomforest': 'Random Forest',
    'gradientboosting': 'Gradient Boosting',
    'gpr': 'Gaussian Process',
    'mlp': 'MLP Neural Network'
}

FEATURES = META.get('features', [
    'log_population', 'log_gdp', 'log_electricity_generation',
    'coal_production', 'oil_production', 'gas_production',
    'renewables_share_elec', 'fossil_share_elec',
    'solar_share_elec', 'wind_share_elec', 'hydro_share_elec',
    'year_norm'
])


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "results": RESULTS,
        "model_display": MODEL_DISPLAY
    })


@app.get("/results", response_class=HTMLResponse)
async def results_page(request: Request):
    best = {}
    if RESULTS:
        for metric in ['RMSE', 'MAE', 'R2', 'MAPE']:
            if metric == 'R2':
                best[metric] = max(RESULTS, key=lambda k: RESULTS[k][metric])
            else:
                best[metric] = min(RESULTS, key=lambda k: RESULTS[k][metric])

    return templates.TemplateResponse("results.html", {
        "request": request,
        "results": RESULTS,
        "best": best,
        "model_display": MODEL_DISPLAY,
        "feature_importance": FEATURE_IMPORTANCE
    })


@app.get("/predict", response_class=HTMLResponse)
async def predict_page(request: Request):
    processed_path = os.path.join(BASE_DIR, '..', 'data', 'processed_data.csv')
    countries = META.get('countries', [])
    if not countries and os.path.exists(processed_path):
        df = pd.read_csv(processed_path)
        countries = sorted(df['country'].unique().tolist())

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "countries": countries,
        "model_display": MODEL_DISPLAY,
        "prediction": None
    })


@app.post("/predict", response_class=HTMLResponse)
async def run_prediction(
    request: Request,
    country: str = Form(...),
    year: int = Form(...),
    population: float = Form(...),
    gdp: float = Form(...),
    electricity_gen: float = Form(...),
    coal_prod: float = Form(0.0),
    oil_prod: float = Form(0.0),
    gas_prod: float = Form(0.0),
    renewables_share: float = Form(...),
    fossil_share: float = Form(...),
    solar_share: float = Form(0.0),
    wind_share: float = Form(0.0),
    hydro_share: float = Form(0.0),
):
    year_norm = (year - 1990) / (2022 - 1990)

    feature_vector = np.array([[
        np.log1p(population),
        np.log1p(gdp),
        np.log1p(electricity_gen),
        coal_prod,
        oil_prod,
        gas_prod,
        renewables_share,
        fossil_share,
        solar_share,
        wind_share,
        hydro_share,
        year_norm
    ]])

    predictions = {}
    for name, model in MODELS.items():
        try:
            pred_log = model.predict(feature_vector)[0]
            pred_twh = round(float(np.expm1(pred_log)), 1)
            predictions[name] = pred_twh
        except Exception as e:
            predictions[name] = f"Error: {e}"

    countries = META.get('countries', [])

    return templates.TemplateResponse("predict.html", {
        "request": request,
        "countries": countries,
        "model_display": MODEL_DISPLAY,
        "prediction": predictions,
        "input_country": country,
        "input_year": year
    })


@app.get("/eda", response_class=HTMLResponse)
async def eda_page(request: Request):
    return templates.TemplateResponse("eda.html", {"request": request})


# ── World Map ─────────────────────────────────────────────────────────────────

@app.get("/map", response_class=HTMLResponse)
async def map_page(request: Request):
    """
    Interactive world map page. The template loads map_data.json client-side
    via fetch() and renders a D3.js choropleth. The sidebar shows static
    overview charts generated by notebooks/03_map_data.ipynb.
    """
    rankings = {}
    rankings_path = os.path.join(BASE_DIR, 'static', 'map_rankings.json')
    if os.path.exists(rankings_path):
        with open(rankings_path) as f:
            rankings = json.load(f)

    map_meta = {}
    meta_path = os.path.join(BASE_DIR, 'static', 'map_meta.json')
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            map_meta = json.load(f)

    return templates.TemplateResponse("map.html", {
        "request": request,
        "rankings": rankings,
        "map_meta": map_meta,
        "data_ready": os.path.exists(os.path.join(BASE_DIR, 'static', 'map_data.json'))
    })


@app.get("/api/map-data/{year}")
async def api_map_data(year: int):
    """
    Returns the map dataset for a single year as JSON.
    The JS client calls this when the user changes the year slider.
    """
    map_data_path = os.path.join(BASE_DIR, 'static', 'map_data.json')
    if not os.path.exists(map_data_path):
        return {"error": "Map data not found. Run notebooks/03_map_data.ipynb first."}

    with open(map_data_path) as f:
        all_data = json.load(f)

    year_data = all_data.get(str(year), {})
    return {"year": year, "data": year_data}


@app.get("/api/country/{iso_code}")
async def api_country_timeseries(iso_code: str):
    """
    Returns the full time series for a single country (all years).
    Called when the user clicks a country on the map.
    """
    map_data_path = os.path.join(BASE_DIR, 'static', 'map_data.json')
    if not os.path.exists(map_data_path):
        return {"error": "Map data not found."}

    with open(map_data_path) as f:
        all_data = json.load(f)

    iso_code = iso_code.upper()
    timeseries = {}
    for year, countries in all_data.items():
        if iso_code in countries:
            timeseries[year] = countries[iso_code]

    if not timeseries:
        return {"error": f"No data found for ISO code: {iso_code}"}

    country_name = next(iter(timeseries.values())).get('country', iso_code)
    return {
        "iso_code": iso_code,
        "country": country_name,
        "timeseries": timeseries
    }


# ── Utility API ───────────────────────────────────────────────────────────────

@app.get("/api/results")
async def api_results():
    return RESULTS


@app.get("/api/feature-importance")
async def api_feature_importance():
    return FEATURE_IMPORTANCE
