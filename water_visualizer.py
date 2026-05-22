"""
water_visualizer.py  —  Week 2B of the Water Quality Intelligence Project
Loads the streamflow CSV and produces 3 saved chart images.


"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates   # for formatting date labels on x-axis
from pathlib import Path


# ── Configuration ─────────────────────────────────────────────────────────────
CSV_FILE   = "streamflow_data.csv"
OUTPUT_DIR = "charts"               # folder where chart images will be saved
# ─────────────────────────────────────────────────────────────────────────────


def load_and_prepare(filepath: str) -> pd.DataFrame:
    """
    Same load logic as water_analyzer.py — reads CSV, parses dates, sets index.
    Also adds the 7-day rolling average column we'll need for Chart 1.
    """
    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df["flow_7day_avg"] = df["flow_cfs"].rolling(window=7, min_periods=1).mean()
    return df


def setup_output_dir(dir_name: str) -> Path:
    """
    Creates the charts/ folder if it doesn't exist.
    Path.mkdir(exist_ok=True) won't throw an error if folder already exists.
    """
    output_path = Path(dir_name)
    output_path.mkdir(exist_ok=True)
    print(f"Charts will be saved to: {output_path.resolve()}\n")
    return output_path


# ── CHART 1: Daily flow + 7-day rolling average ───────────────────────────────
def chart_daily_flow(df: pd.DataFrame, output_dir: Path) -> None:
    """
    A line chart showing raw daily flow and the smoothed 7-day average.

    MATPLOTLIB STRUCTURE — understand this and everything else makes sense:
      fig  = the entire figure (like a canvas)
      ax   = the axes (the actual plot area with x/y axes)

    fig, ax = plt.subplots() creates both at once.
    Everything you draw goes on ax — ax.plot(), ax.set_title(), etc.
    """
    fig, ax = plt.subplots(figsize=(12, 5))   # width=12 inches, height=5 inches

    # Plot raw daily flow — thin, light blue, slightly transparent
    ax.plot(
        df.index,           # x axis: dates
        df["flow_cfs"],     # y axis: flow values
        color="#90CAF9",    # light blue
        linewidth=0.8,
        alpha=0.7,          # 70% opacity — lets the average line show through
        label="Daily flow (cfs)",
    )

    # Plot 7-day rolling average — thicker, darker blue
    ax.plot(
        df.index,
        df["flow_7day_avg"],
        color="#1565C0",    # dark blue
        linewidth=2,
        label="7-day average",
    )

    # ── Annotate the peak spike ───────────────────────────────────────────────
    # Find the date and value of the maximum flow
    peak_date  = df["flow_cfs"].idxmax()
    peak_value = df["flow_cfs"].max()

    # ax.annotate() draws an arrow + text label at a specific point
    ax.annotate(
        f"Peak: {peak_value:.1f} cfs\n{peak_date.strftime('%b %d')}",
        xy=(peak_date, peak_value),             # arrow tip points here
        xytext=(peak_date, peak_value + 3),     # text appears above the tip
        fontsize=9,
        color="#B71C1C",
        arrowprops=dict(arrowstyle="->", color="#B71C1C"),
        ha="center",
    )

    # ── Add a low-flow threshold line ─────────────────────────────────────────
    low_flow_threshold = 0.5   # cfs — below this is notably dry for this creek
    ax.axhline(
        y=low_flow_threshold,
        color="#FF8F00",
        linewidth=1,
        linestyle="--",
        label=f"Low-flow threshold ({low_flow_threshold} cfs)",
    )

    # ── Labels, title, formatting ─────────────────────────────────────────────
    ax.set_title(
        "Little Hope Creek — Daily Streamflow (Charlotte, NC)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Flow (cfs)", fontsize=10)

    # Format x-axis to show month names instead of raw date numbers
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")

    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)   # subtle horizontal gridlines

    # tight_layout() prevents labels from being cut off at the edges
    plt.tight_layout()

    filepath = output_dir / "chart1_daily_flow.png"
    plt.savefig(filepath, dpi=150)     # dpi=150 gives a crisp image
    plt.close()                        # close the figure to free memory
    print(f"Saved: {filepath}")


# ── CHART 2: Monthly average bar chart ───────────────────────────────────────
def chart_monthly_averages(df: pd.DataFrame, output_dir: Path) -> None:
    """
    A bar chart showing average flow per month.

    .resample("ME").mean() groups all daily values by month
    and calculates the mean for each group — same as Week 2A.

    Bar charts are better than line charts here because months
    are discrete categories, not a continuous time series.
    """
    # Resample to monthly averages
    monthly = df["flow_cfs"].resample("ME").mean().round(2)

    # Format index as readable month labels: "Jan 2026", "Feb 2026" etc
    labels = monthly.index.strftime("%b %Y")

    # Color bars by value — higher flow = darker blue
    # We normalize each value between 0 and 1, then map to a colormap
    norm_values = (monthly - monthly.min()) / (monthly.max() - monthly.min())
    colors = plt.cm.Blues(0.3 + norm_values * 0.6)   # range from light to dark blue

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(labels, monthly, color=colors, edgecolor="white", linewidth=0.5)

    # Add value labels on top of each bar
    for bar, value in zip(bars, monthly):
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # center of bar
            bar.get_height() + 0.05,              # just above the bar
            f"{value:.2f}",
            ha="center", va="bottom",
            fontsize=9, color="#333333"
        )

    ax.set_title(
        "Little Hope Creek — Monthly Average Streamflow",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Month", fontsize=10)
    ax.set_ylabel("Average Flow (cfs)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    filepath = output_dir / "chart2_monthly_averages.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


# ── CHART 3: Spike detection overlay ─────────────────────────────────────────
def chart_spike_detection(df: pd.DataFrame, output_dir: Path) -> None:
    """
    Shows the daily flow with spike days highlighted as red dots.

    This is a preview of Week 3 — visually communicating which days
    were anomalous using the 2-sigma threshold from the analyzer.

    New technique: ax.scatter() plots individual points on top of a line.
    """
    mean_flow = df["flow_cfs"].mean()
    std_flow  = df["flow_cfs"].std()
    threshold = mean_flow + (2 * std_flow)

    # Boolean mask to find spike rows
    spike_mask = df["flow_cfs"] > threshold
    spikes     = df[spike_mask]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Base line — daily flow in grey
    ax.plot(
        df.index,
        df["flow_cfs"],
        color="#90A4AE",
        linewidth=1,
        label="Daily flow (cfs)",
    )

    # Threshold line — dashed red
    ax.axhline(
        y=threshold,
        color="#E53935",
        linewidth=1.2,
        linestyle="--",
        label=f"Spike threshold ({threshold:.1f} cfs)",
    )

    # Scatter plot — red dots on spike days
    # ax.scatter() is like ax.plot() but draws individual points, not a line
    ax.scatter(
        spikes.index,
        spikes["flow_cfs"],
        color="#E53935",
        zorder=5,           # zorder controls layering — higher = drawn on top
        s=60,               # dot size
        label=f"Spike days ({len(spikes)} events)",
    )

    # Label each spike dot with its date
    for date, row in spikes.iterrows():
        ax.annotate(
            date.strftime("%b %d"),
            xy=(date, row["flow_cfs"]),
            xytext=(0, 8),                          # offset 8 points upward
            textcoords="offset points",
            fontsize=7.5,
            ha="center",
            color="#B71C1C",
        )

    ax.set_title(
        "Little Hope Creek — Streamflow Anomaly Detection (2σ threshold)",
        fontsize=13, fontweight="bold", pad=12
    )
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Flow (cfs)", fontsize=10)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=30, ha="right")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    filepath = output_dir / "chart3_spike_detection.png"
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"Saved: {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df         = load_and_prepare(CSV_FILE)
    output_dir = setup_output_dir(OUTPUT_DIR)

    print("Generating charts...\n")
    chart_daily_flow(df, output_dir)
    chart_monthly_averages(df, output_dir)
    chart_spike_detection(df, output_dir)

    print("\nAll done! Open the 'charts' folder to see your images.")
    print("These go straight into your project README on GitHub.")
