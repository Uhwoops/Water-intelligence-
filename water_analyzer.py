"""
water_analyzer.py  —  Water Quality Intelligence Pipeline
Loads streamflow CSV into pandas and performs time-series analysis.
"""

import pandas as pd
from pathlib import Path


CSV_FILE = "streamflow_data.csv"


def load_data(filepath: str) -> pd.DataFrame:
    """Reads the CSV, parses dates, and sets the date column as the index."""
    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"Could not find '{filepath}'.\n"
            f"Make sure water_fetcher.py ran successfully first."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.set_index("date")
    df = df.sort_index()
    return df


def explore(df: pd.DataFrame) -> None:
    """Prints a structured overview of the DataFrame."""
    print("══ DATAFRAME OVERVIEW ══════════════════════════════")

    rows, cols = df.shape
    print(f"  Shape      : {rows} rows x {cols} columns")
    print(f"  Date range : {df.index.min().date()}  to  {df.index.max().date()}")
    print(f"  Station    : {df['site_name'].iloc[0]}")

    print("\n── Column types ────────────────────────────────────")
    print(df.dtypes)

    print("\n── Missing values ──────────────────────────────────")
    print(df.isnull().sum())

    print("\n── First 5 rows ────────────────────────────────────")
    print(df.head())

    print("\n── Last 5 rows ─────────────────────────────────────")
    print(df.tail())
    print("════════════════════════════════════════════════════\n")


def basic_stats(df: pd.DataFrame) -> None:
    """Calculates summary statistics on the flow_cfs column."""
    print("══ BASIC STATISTICS ════════════════════════════════")

    flow = df["flow_cfs"]

    print("\n── .describe() ─────────────────────────────────────")
    print(flow.describe().round(2))

    print("\n── Individual calculations ─────────────────────────")
    print(f"  Mean flow    : {flow.mean():.2f} cfs")
    print(f"  Median flow  : {flow.median():.2f} cfs")
    print(f"  Std deviation: {flow.std():.2f} cfs")
    print(f"  Peak flow    : {flow.max():.2f} cfs  on {flow.idxmax().date()}")
    print(f"  Lowest flow  : {flow.min():.2f} cfs  on {flow.idxmin().date()}")
    print("════════════════════════════════════════════════════\n")


def monthly_summary(df: pd.DataFrame) -> None:
    """Groups data by month and calculates stats per month."""
    print("══ MONTHLY SUMMARY ═════════════════════════════════")

    monthly = (
        df["flow_cfs"]
        .resample("ME")
        .agg(
            mean_flow="mean",
            max_flow="max",
            min_flow="min",
            days_of_data="count"
        )
        .round(2)
    )

    monthly.index = monthly.index.strftime("%b %Y")
    print(monthly.to_string())
    print("════════════════════════════════════════════════════\n")


def rolling_average(df: pd.DataFrame) -> pd.DataFrame:
    """Adds a 7-day rolling average column to smooth day-to-day noise."""
    print("══ ROLLING AVERAGE ═════════════════════════════════")

    df["flow_7day_avg"] = (
        df["flow_cfs"]
        .rolling(window=7, min_periods=1)
        .mean()
        .round(3)
    )

    print("  Added column: flow_7day_avg (7-day rolling mean)")
    print("\n── Sample: raw flow vs smoothed average ────────────")
    print(df[["flow_cfs", "flow_7day_avg"]].head(14).to_string())
    print("════════════════════════════════════════════════════\n")
    return df


def find_spikes(df: pd.DataFrame) -> None:
    """Flags days where flow exceeded mean + 2 standard deviations."""
    print("══ SPIKE DETECTION ═════════════════════════════════")

    flow      = df["flow_cfs"]
    mean_flow = flow.mean()
    std_flow  = flow.std()
    threshold = mean_flow + (2 * std_flow)

    print(f"  Mean flow : {mean_flow:.2f} cfs")
    print(f"  Std dev   : {std_flow:.2f} cfs")
    print(f"  Threshold : {threshold:.2f} cfs  (mean + 2 x std)")

    spike_mask = flow > threshold
    spikes     = df[spike_mask]["flow_cfs"]

    print(f"\n  Days above threshold: {len(spikes)}")

    if len(spikes) > 0:
        print("\n── Spike days ──────────────────────────────────────")
        for date, value in spikes.items():
            print(f"  {date.date()}  :  {value:.2f} cfs")

    print("════════════════════════════════════════════════════\n")


if __name__ == "__main__":
    print("\nLoading data...\n")
    df = load_data(CSV_FILE)

    explore(df)
    basic_stats(df)
    monthly_summary(df)
    df = rolling_average(df)
    find_spikes(df)

    print("Analysis complete. Run water_visualizer.py to see the charts.")
