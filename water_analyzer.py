"""
water_analyzer.py  —  Week 2A of the Water Quality Intelligence Project
Loads the streamflow CSV into pandas and explores the data.

"""

import pandas as pd   
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE = "streamflow_data.csv"
# ─────────────────────────────────────────────────────────────────────────────


def load_data(filepath: str) -> pd.DataFrame:
   
    if not Path(filepath).exists():
        raise FileNotFoundError(
            f"Could not find '{filepath}'.\n"
            f"Make sure water_fetcher.py ran successfully first."
        )

    df = pd.read_csv(
        filepath,
        parse_dates=["date"],   
    )

    
    df = df.set_index("date")

    df = df.sort_index()

    return df


def explore(df: pd.DataFrame) -> None:
    """
    Prints a structured overview of the DataFrame.
    """
    print("══ DATAFRAME OVERVIEW ══════════════════════════════")


    rows, cols = df.shape
    print(f"  Shape      : {rows} rows × {cols} columns")

   
    print(f"  Date range : {df.index.min().date()}  →  {df.index.max().date()}")
    print(f"  Station    : {df['site_name'].iloc[0]}")

    print("\n── Column types (dtypes) ───────────────────────────")
    # dtypes tells you what kind of data is in each column
    # float64 = decimal number, object = string, datetime = date
    print(df.dtypes)

    print("\n── Missing values per column ───────────────────────")
    print(df.isnull().sum())

    print("\n── First 5 rows (.head()) ──────────────────────────")
    print(df.head())

    print("\n── Last 5 rows (.tail()) ───────────────────────────")
    print(df.tail())

    print("════════════════════════════════════════════════════\n")


def basic_stats(df: pd.DataFrame) -> None:
   
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
  
    print("══ SPIKE DETECTION ═════════════════════════════════")

    flow      = df["flow_cfs"]
    mean_flow = flow.mean()
    std_flow  = flow.std()
    threshold = mean_flow + (2 * std_flow)

    print(f"  Mean flow : {mean_flow:.2f} cfs")
    print(f"  Std dev   : {std_flow:.2f} cfs")
    print(f"  Threshold : {threshold:.2f} cfs  (mean + 2×std)")

   
    spike_mask = flow > threshold         
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
