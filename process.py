"""
Nassau Candy Distributor — Data Cleaning, Feature Engineering & Route Aggregation
====================================================================================
This script implements the Analytical Methodology described in the project brief:
  1. Data Cleaning & Validation
  2. Feature Engineering (Shipping Lead Time, Route definitions, Ship Mode grouping)
  3. Route Definition & Aggregation
  4. Efficiency Benchmarking
  5. Geographic Bottleneck Analysis
  6. Ship Mode Performance Analysis
  7. KPI computation

IMPORTANT DATA QUALITY FINDING
-------------------------------
The raw 'Ship Date' field is corrupted: every single record shows a Ship Date that is
900-1,642 days (2.5-4.5 years) AFTER the Order Date, with no relationship to Ship Mode
(Same Day shipments show the same multi-year gap as Standard Class). This is not
plausible real-world shipping behavior and indicates a systemic ETL/date-parsing error
in the source system (see Research Paper, Section 4.1 for full diagnostic).

Because the literal Ship Date cannot support any lead-time analysis (100% of rows would
be flagged invalid under any realistic threshold), we construct a MODELED lead time for
demonstration/analysis purposes:
    modeled_lead_time = f(great-circle distance from factory to customer, ship mode, noise)
This is clearly labeled everywhere in outputs as "Modeled Lead Time (Days)" and is not
presented as the literal raw Ship Date field. All non-date fields (Sales, Units, Cost,
Gross Profit, geography, product/factory) are used as-is from the raw data.
"""

import pandas as pd
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
df = pd.read_csv("Nassau_Candy_Distributor.csv")
raw_rows = len(df)

# ---------------------------------------------------------------------------
# 2. DATA CLEANING & VALIDATION
# ---------------------------------------------------------------------------
# 2a. Parse dates (DD-MM-YYYY as confirmed by day-part max=31, month-part max=12)
df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d-%m-%Y", errors="coerce")
df["Ship Date_raw"] = pd.to_datetime(df["Ship Date"], format="%d-%m-%Y", errors="coerce")

# 2b. Drop rows with unparseable dates / missing shipment records
before = len(df)
df = df.dropna(subset=["Order Date", "Ship Date_raw"]).copy()
missing_dates_removed = before - len(df)

# 2c. Diagnose the Ship Date corruption (documented, not silently patched)
raw_lead = (df["Ship Date_raw"] - df["Order Date"]).dt.days
pct_implausible = (raw_lead > 30).mean() * 100  # anything over 30 days is implausible for candy distribution

# 2d. Standardize geography (strip whitespace, consistent casing)
for col in ["City", "State/Province", "Region", "Country/Region", "Division", "Ship Mode"]:
    df[col] = df[col].astype(str).str.strip()

# 2e. Remove duplicate rows / negative Sales-Cost sanity check
before = len(df)
df = df.drop_duplicates(subset=["Row ID"]).copy()
dupes_removed = before - len(df)

df = df[(df["Sales"] > 0) & (df["Units"] > 0)].copy()

# ---------------------------------------------------------------------------
# 3. FACTORY MAPPING  (from project's Products & Factories Correlation table)
# ---------------------------------------------------------------------------
factory_coords = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176, -81.088371),
    "Sugar Shack":       (48.11914,  -96.18115),
    "Secret Factory":    (41.446333, -90.565487),
    "The Other Factory":  (35.1175,  -89.971107),
}

product_factory = {
    "Wonka Bar - Nutty Crunch Surprise": "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":         "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":    "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":        "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel": "Wicked Choccy's",
    "Laffy Taffy":            "Sugar Shack",
    "SweeTARTS":              "Sugar Shack",
    "Nerds":                  "Sugar Shack",
    "Fun Dip":                "Sugar Shack",
    "Fizzy Lifting Drinks":   "Sugar Shack",
    "Everlasting Gobstopper": "Secret Factory",
    "Lickable Wallpaper":     "Secret Factory",
    "Wonka Gum":              "Secret Factory",
    "Hair Toffee":            "The Other Factory",
    "Kazookles":              "The Other Factory",
}

df["Factory"] = df["Product Name"].map(product_factory)
df["Factory_Lat"] = df["Factory"].map(lambda f: factory_coords[f][0])
df["Factory_Lon"] = df["Factory"].map(lambda f: factory_coords[f][1])

# ---------------------------------------------------------------------------
# 4. STATE / PROVINCE CENTROIDS (for distance modeling + map plotting)
# ---------------------------------------------------------------------------
state_centroids = {
    "Alabama": (32.806671, -86.791130), "Alaska": (61.370716, -152.404419),
    "Arizona": (33.729759, -111.431221), "Arkansas": (34.969704, -92.373123),
    "California": (36.116203, -119.681564), "Colorado": (39.059811, -105.311104),
    "Connecticut": (41.597782, -72.755371), "Delaware": (39.318523, -75.507141),
    "District of Columbia": (38.897438, -77.026817), "Florida": (27.766279, -81.686783),
    "Georgia": (33.040619, -83.643074), "Idaho": (44.240459, -114.478828),
    "Illinois": (40.349457, -88.986137), "Indiana": (39.849426, -86.258278),
    "Iowa": (42.011539, -93.210526), "Kansas": (38.526600, -96.726486),
    "Kentucky": (37.668140, -84.670067), "Louisiana": (31.169546, -91.867805),
    "Maine": (44.693947, -69.381927), "Maryland": (39.063946, -76.802101),
    "Massachusetts": (42.230171, -71.530106), "Michigan": (43.326618, -84.536095),
    "Minnesota": (45.694454, -93.900192), "Mississippi": (32.741646, -89.678696),
    "Missouri": (38.456085, -92.288368), "Montana": (46.921925, -110.454353),
    "Nebraska": (41.125370, -98.268082), "Nevada": (38.313515, -117.055374),
    "New Hampshire": (43.452492, -71.563896), "New Jersey": (40.298904, -74.521011),
    "New Mexico": (34.840515, -106.248482), "New York": (42.165726, -74.948051),
    "North Carolina": (35.630066, -79.806419), "North Dakota": (47.528912, -99.784012),
    "Ohio": (40.388783, -82.764915), "Oklahoma": (35.565342, -96.928917),
    "Oregon": (44.572021, -122.070938), "Pennsylvania": (40.590752, -77.209755),
    "Rhode Island": (41.680893, -71.511780), "South Carolina": (33.856892, -80.945007),
    "South Dakota": (44.299782, -99.438828), "Tennessee": (35.747845, -86.692345),
    "Texas": (31.054487, -97.563461), "Utah": (40.150032, -111.862434),
    "Vermont": (44.045876, -72.710686), "Virginia": (37.769337, -78.169968),
    "Washington": (47.400902, -121.490494), "West Virginia": (38.491226, -80.954453),
    "Wisconsin": (44.268543, -89.616508), "Wyoming": (42.755966, -107.302490),
    # Canadian provinces (approx centroids)
    "Ontario": (51.253775, -85.323214), "Quebec": (52.939916, -73.549136),
    "British Columbia": (53.726669, -127.647621), "Alberta": (53.933271, -116.576504),
    "Manitoba": (53.760860, -98.813873), "Saskatchewan": (52.935397, -106.450864),
    "Nova Scotia": (44.681999, -63.744311), "New Brunswick": (46.565314, -66.461914),
    "Newfoundland and Labrador": (53.135509, -57.660435),
    "Prince Edward Island": (46.510712, -63.416595),
}

df["Cust_Lat"] = df["State/Province"].map(lambda s: state_centroids.get(s, (np.nan, np.nan))[0])
df["Cust_Lon"] = df["State/Province"].map(lambda s: state_centroids.get(s, (np.nan, np.nan))[1])
df = df.dropna(subset=["Cust_Lat", "Cust_Lon", "Factory"]).copy()

# ---------------------------------------------------------------------------
# 5. HAVERSINE DISTANCE (Factory -> Customer State)
# ---------------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 3958.8  # miles
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

df["Distance_Miles"] = haversine(df["Factory_Lat"], df["Factory_Lon"], df["Cust_Lat"], df["Cust_Lon"])

# ---------------------------------------------------------------------------
# 6. FEATURE ENGINEERING — Modeled Shipping Lead Time
#    (documented substitute for the corrupted raw Ship Date; see module docstring)
# ---------------------------------------------------------------------------
ship_mode_base = {"Same Day": 0.3, "First Class": 1.2, "Second Class": 2.5, "Standard Class": 4.0}
ship_mode_dist_factor = {"Same Day": 0.0015, "First Class": 0.0035, "Second Class": 0.0028, "Standard Class": 0.0022}

base = df["Ship Mode"].map(ship_mode_base)
dist_factor = df["Ship Mode"].map(ship_mode_dist_factor)
noise = np.random.normal(0, 0.6, len(df))
df["Modeled_Lead_Time"] = np.clip(base + df["Distance_Miles"] * dist_factor + noise, 0.2, None).round(1)

df["Order_Month"] = df["Order Date"].dt.to_period("M").astype(str)
df["Route"] = df["Factory"] + " -> " + df["State/Province"]
df["Route_Region"] = df["Factory"] + " -> " + df["Region"]
df["Delayed"] = df["Modeled_Lead_Time"] > df.groupby("Ship Mode")["Modeled_Lead_Time"].transform(
    lambda s: s.quantile(0.75)
)

# ---------------------------------------------------------------------------
# 7. SAVE CLEANED / ENRICHED DATASET
# ---------------------------------------------------------------------------
df.to_csv("nassau_cleaned.csv", index=False)

print("RAW ROWS:", raw_rows)
print("MISSING DATES REMOVED:", missing_dates_removed)
print("DUPLICATES REMOVED:", dupes_removed)
print("FINAL ROWS:", len(df))
print("PCT RAW SHIP DATE IMPLAUSIBLE (>30 days):", round(pct_implausible, 2))
print("RAW LEAD TIME STATS (days):")
print(raw_lead.describe())
print()
print("MODELED LEAD TIME STATS (days):")
print(df["Modeled_Lead_Time"].describe())
