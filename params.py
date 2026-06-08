from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"

RAW_FILENAME = "eua_carbon_futures_daily.csv"
PROCESSED_FILENAME = "eua_daily_returns.csv"

RAW_PATH = DATA_RAW / RAW_FILENAME
PROCESSED_PATH = DATA_PROCESSED / PROCESSED_FILENAME

# Raw data schema
RAW_COLUMNS = {
    "date": "Date",
    "close": "Price",
}

# Investing.com uses month/day/year
RAW_DATE_FORMAT = "%m/%d/%Y"

# cleaning
START_DATE = "2008-01-01"
END_DATE = None  # use the most recent date in the file

# sanity bounds on the EUA price in euros per tonne
# values outside this range treated as data errors and dropped
PRICE_MIN = 0.01
PRICE_MAX = 250.0

RETURN_FLAG_THRESHOLD = 0.5

# regime model
N_REGIMES_GRID = [2, 3, 4]
N_HMM_INITS = 20
HMM_MAX_ITER = 500
RANDOM_SEED = 467

# empirical counting baseline
VOL_WINDOW = 21
VOL_LOW_QUANTILE = 0.33
VOL_HIGH_QUANTILE = 0.66

# EU ETS trading phases for shading
# end set to None means "to the end of the sample"
EU_ETS_PHASES = [
    ("Phase II", "2008-01-01", "2012-12-31"),
    ("Phase III", "2013-01-01", "2020-12-31"),
    ("Phase IV", "2021-01-01", None),
]

# plot style
FIGURE_DPI = 150
FIGURE_SIZE_WIDE = (11, 4)
FIGURE_SIZE_SQUARE = (6, 5)
REGIME_COLORS = ["#4C72B0", "#DD8452", "#C44E52", "#8172B3"]
