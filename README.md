# Nassau Candy — Shipping Route Efficiency Dashboard

## Run it
```
pip install -r requirements.txt
streamlit run app.py
```
Then open the local URL Streamlit prints (usually http://localhost:8501).

## Files
- `app.py` — the Streamlit dashboard (4 tabs: Route Efficiency Overview, Geographic Shipping Map, Ship Mode Comparison, Route Drill-Down)
- `nassau_cleaned.csv` — cleaned & feature-engineered dataset (output of process.py) that app.py reads
- `process.py` (see main deliverables folder) — regenerates nassau_cleaned.csv from the raw source CSV if needed

## Data quality note
The raw source 'Ship Date' field is corrupted (see Research Paper §4.2). The dashboard
uses a documented **Modeled Lead Time** (distance + ship mode based) instead of the raw,
unusable field. This is disclosed in the dashboard sidebar.
