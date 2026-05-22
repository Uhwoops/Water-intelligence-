"""
water_analyzer.py  —  Week 2A of the Water Quality Intelligence Project
Loads the streamflow CSV into pandas and explores the data.

New concepts this week:
  - pandas DataFrames (like a programmable spreadsheet)
  - Reading CSV files with pd.read_csv()
  - Selecting columns and filtering rows
  - DateTime indexing (treating dates as the row label)
  - Aggregations: mean, max, min, std
  - Rolling averages (smoothing out noise)
  - Boolean masks (filtering by condition)
"""

import pandas as pd   # pd is the universal alias — every pandas developer uses it
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE = "streamflow_data.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
    """
    Reads the CSV into a pandas DataFrame and prepares it for analysis.

    A DataFrame is like a spreadsheet with superpowers:
      - Rows are observations (one per day)
      - Columns are variables (date, flow_cfs, site_name, station_id)
      - You can filter, sort, group, and calculate across it in one line

    Key steps here:
      1. pd.read_csv() reads the file into a DataFrame
      2. parse_dates tells pandas the 'date' column is a date, not a string
      3. set_index() makes the date the row label instead of 0, 1, 2...
      4. sort_index() orders rows oldest → newest
    """
    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"Could not find '{filepath}'.\n"
            f"Make sure water_fetcher.py ran successfully first."
        )

    df = pd.read_csv(
        filepath,
        parse_dates=["date"],   # convert the date column from string to datetime
    )

    # Set the date column as the index (row label)
    # This unlocks time-series features like .resample() later
    df = df.set_index("date")

    # Sort oldest to newest — API data isn't always in order
    df = df.sort_index()

    return df


def explore(df: pd.DataFrame) -> None:
    """
    Prints a structured overview of the DataFrame.

    These are the first things any data engineer checks when
    they see a new dataset — shape, types, nulls, basic stats.
    """
    print("══ DATAFRAME OVERVIEW ══════════════════════════════")

    # .shape returns (rows, columns) as a tuple
    rows, cols = df.shape
    print(f"  Shape      : {rows} rows × {cols} columns")

    # .index gives you the row labels — here they're dates
    print(f"  Date range : {df.index.min().date()}  →  {df.index.max().date()}")
    print(f"  Station    : {df['site_name'].iloc[0]}")

    print("\n── Column types (dtypes) ───────────────────────────")
    # dtypes tells you what kind of data is in each column
    # float64 = decimal number, object = string, datetime = date
    print(df.dtypes)

    print("\n── Missing values per column ───────────────────────")
    # .isnull().sum() counts the None/NaN values in each column
    # NaN = "Not a Number" — pandas way of representing missing data
    print(df.isnull().sum())

    print("\n── First 5 rows (.head()) ──────────────────────────")
    # .head() shows the first N rows — the fastest way to sanity-check your data
    print(df.head())

    print("\n── Last 5 rows (.tail()) ───────────────────────────")
    print(df.tail())

    print("════════════════════════════════════════════════════\n")


def basic_stats(df: pd.DataFrame) -> None:
    """
    Calculates summary statistics on the flow_cfs column.

    .describe() is the single most useful pandas method for exploration —
    it gives you count, mean, std, min, quartiles, and max in one call.
    """
    print("══ BASIC STATISTICS ════════════════════════════════")

    flow = df["flow_cfs"]   # select a single column → returns a Series

    # .describe() gives you a full statistical summary instantly
    print("\n── .describe() ─────────────────────────────────────")
    print(flow.describe().round(2))

    # Individual stats
    print("\n── Individual calculations ─────────────────────────")
    print(f"  Mean flow    : {flow.mean():.2f} cfs")
    print(f"  Median flow  : {flow.median():.2f} cfs")
    print(f"  Std deviation: {flow.std():.2f} cfs")
    print(f"  Peak flow    : {flow.max():.2f} cfs  on {flow.idxmax().date()}")
    print(f"  Lowest flow  : {flow.min():.2f} cfs  on {flow.idxmin().date()}")

    # idxmax() / idxmin() return the INDEX LABEL of the max/min value
    # Since our index is dates, this tells us exactly which day had the peak

    print("════════════════════════════════════════════════════\n")


def monthly_summary(df: pd.DataFrame) -> None:
    """
    Groups data by month and calculates stats per month.

    .resample("ME") is like a GROUP BY for time series.
    "ME" means Month End — it groups all rows in each calendar month together.
    Then .agg() lets you calculate multiple stats at once.

    This is one of the most powerful pandas patterns for time series data.
    """
    print("══ MONTHLY SUMMARY ═════════════════════════════════")

    monthly = (
        df["flow_cfs"]
        .resample("ME")           # group by month
        .agg(                     # calculate multiple stats per group
            mean_flow="mean",
            max_flow="max",
            min_flow="min",
            days_of_data="count"
        )
        .round(2)
    )

    # Format the index to show "Jan 2026" instead of "2026-01-31"
    monthly.index = monthly.index.strftime("%b %Y")

    print(monthly.to_string())
    print("════════════════════════════════════════════════════\n")


def rolling_average(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 7-day rolling average column to the DataFrame.

    A rolling average smooths out day-to-day noise so you can
    see the underlying trend. It works by averaging the current
    row with the N-1 rows before it.

    Example with window=3:
      Day 1: 1.0  → no average yet (not enough data)
      Day 2: 2.0  → no average yet
      Day 3: 3.0  → average of [1, 2, 3] = 2.0
      Day 4: 4.0  → average of [2, 3, 4] = 3.0

    min_periods=1 means: start averaging as soon as you have at least
    1 value, instead of waiting for a full 7-day window.
    """
    print("══ ROLLING AVERAGE ═════════════════════════════════")

    df["flow_7day_avg"] = (
        df["flow_cfs"]
        .rolling(window=7, min_periods=1)
        .mean()
        .round(3)
    )

    print("  Added column: flow_7day_avg (7-day rolling mean)")
    print("\n── Sample: raw flow vs smoothed average ────────────")

    # Show both columns side by side to see the smoothing effect
    comparison = df[["flow_cfs", "flow_7day_avg"]].head(14)
    print(comparison.to_string())

    print("════════════════════════════════════════════════════\n")
    return df


def find_spikes(df: pd.DataFrame) -> None:
    """
    Finds days where flow was significantly above normal.

    This uses a BOOLEAN MASK — one of the most important pandas patterns.

    A boolean mask is a Series of True/False values, one per row.
    When you pass it into df[mask], pandas returns only the rows
    where the mask is True.

    We define a spike as: flow > mean + (2 × standard deviation)
    This is called a "2-sigma threshold" — statistically, only ~5%
    of normal readings should exceed it. Anything above is notable.
    """
    print("══ SPIKE DETECTION ═════════════════════════════════")

    flow      = df["flow_cfs"]
    mean_flow = flow.mean()
    std_flow  = flow.std()
    threshold = mean_flow + (2 * std_flow)

    print(f"  Mean flow : {mean_flow:.2f} cfs")
    print(f"  Std dev   : {std_flow:.2f} cfs")
    print(f"  Threshold : {threshold:.2f} cfs  (mean + 2×std)")

    # Build the boolean mask
    spike_mask = flow > threshold           # Series of True/False
    spikes     = df[spike_mask]["flow_cfs"] # filter to only spike rows

    print(f"\n  Days above threshold: {len(spikes)}")

    if len(spikes) > 0:
        print("\n── Spike days ──────────────────────────────────────")
        for date, value in spikes.items():
            print(f"  {date.date()}  →  {value:.2f} cfs")

    print("════════════════════════════════════════════════════\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading data...\n")
    df = load_data(CSV_FILE)

    explore(df)
    basic_stats(df)
    monthly_summary(df)
    df = rolling_average(df)
    find_spikes(df)

    print("Analysis complete. Run water_visualizer.py in Week 2B to see the charts.")
