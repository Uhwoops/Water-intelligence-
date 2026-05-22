"""
water_monitor.py  —  Week 3 of the Water Quality Intelligence Project
Builds a WaterQualityMonitor class with ML-based anomaly detection.

import pandas as pd
import numpy as np
import logging

from datetime import datetime
from pathlib import Path
from sklearn.ensemble import IsolationForest



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),                          # prints to terminal
        logging.FileHandler("water_monitor.log"),         # writes to file
    ]
)

logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────────────────────────────────────


class WaterQualityMonitor:
    """
    A class that loads streamflow data, detects anomalies, and generates reports.

    WHY A CLASS?
    So far your project has three separate scripts that each load the CSV
    and repeat the same setup code. A class solves this by keeping everything
    in one place:

      monitor = WaterQualityMonitor("streamflow_data.csv")
      monitor.detect_anomalies()
      monitor.generate_report()

    That's clean, reusable, and easy to extend.

    CLASS ANATOMY:
      __init__()  = the constructor — runs automatically when you create an object
      self        = refers to THIS specific instance of the class
      self.df     = an instance variable — data stored on the object itself
    """

    def __init__(self, csv_filepath: str):
        """
        Constructor — runs when you do: monitor = WaterQualityMonitor("file.csv")

        Sets up the object by:
          1. Storing the filepath for reference
          2. Loading and preparing the data
          3. Initializing empty containers for results
          4. Logging that the monitor was created
        """
        self.filepath    = csv_filepath
        self.df          = self._load_data()         # underscore = "private" method
        self.anomalies   = pd.DataFrame()            # empty for now, filled later
        self.model       = None                      # ML model, fitted later
        self.report_path = "water_quality_report.txt"

        logger.info(f"WaterQualityMonitor initialized — {len(self.df)} days loaded")
        logger.info(f"Station: {self.df['site_name'].iloc[0]}")
        logger.info(f"Date range: {self.df.index.min().date()} → {self.df.index.max().date()}")


    def _load_data(self) -> pd.DataFrame:
        """
        Private method — the underscore prefix is a convention meaning
        "this is internal, don't call it from outside the class directly."

        Loads and prepares the CSV exactly like our previous scripts,
        but now it lives inside the class so every method can use self.df.
        """
        if not Path(self.filepath).exists():
            raise FileNotFoundError(f"Cannot find: {self.filepath}")

        df = pd.read_csv(self.filepath, parse_dates=["date"])
        df = df.set_index("date").sort_index()

        # Add engineered features — extra columns that help the ML model
        # Rolling stats give the model context: is today unusual vs recent history?
        df["flow_7day_avg"]  = df["flow_cfs"].rolling(window=7,  min_periods=1).mean()
        df["flow_30day_avg"] = df["flow_cfs"].rolling(window=30, min_periods=1).mean()
        df["flow_7day_std"]  = df["flow_cfs"].rolling(window=7,  min_periods=1).std().fillna(0)

        # Ratio of today's flow to the 7-day average
        # A ratio of 5.0 means today was 5x the recent average — suspicious
        df["flow_ratio"] = (df["flow_cfs"] / df["flow_7day_avg"].replace(0, np.nan)).fillna(1)

        return df


    def compute_statistics(self) -> dict:
        """
        Calculates and returns a dictionary of summary statistics.

        Returning a dict (instead of just printing) makes this reusable —
        the generate_report() method can call this and write the results to a file.
        """
        flow  = self.df["flow_cfs"]
        stats = {
            "station":       self.df["site_name"].iloc[0],
            "date_start":    str(self.df.index.min().date()),
            "date_end":      str(self.df.index.max().date()),
            "days_total":    len(flow),
            "mean_flow":     round(float(flow.mean()), 3),
            "median_flow":   round(float(flow.median()), 3),
            "std_flow":      round(float(flow.std()), 3),
            "max_flow":      round(float(flow.max()), 3),
            "max_flow_date": str(flow.idxmax().date()),
            "min_flow":      round(float(flow.min()), 3),
            "min_flow_date": str(flow.idxmin().date()),
            # Percentage of days below 0.5 cfs (low flow / dry conditions)
            "pct_low_flow":  round(float((flow < 0.5).sum() / len(flow) * 100), 1),
        }
        logger.info("Statistics computed")
        return stats


    def detect_anomalies_statistical(self) -> pd.DataFrame:
        """
        Method 1: Statistical anomaly detection using the 2-sigma threshold.
        Same logic as water_analyzer.py — but now it's a proper class method.

        Returns a DataFrame of anomalous rows so we can compare with the ML method.
        """
        flow      = self.df["flow_cfs"]
        threshold = flow.mean() + (2 * flow.std())
        mask      = flow > threshold
        anomalies = self.df[mask][["flow_cfs"]].copy()
        anomalies["method"] = "statistical_2sigma"

        logger.info(f"Statistical detection: {len(anomalies)} anomalies (threshold={threshold:.2f} cfs)")
        return anomalies


    def detect_anomalies_ml(self, contamination: float = 0.05) -> pd.DataFrame:
        """
        Method 2: ML anomaly detection using Isolation Forest.

        HOW ISOLATION FOREST WORKS:
        Imagine randomly drawing a line through your data to isolate one point.
        Normal points (clustered together) need MANY cuts to isolate.
        Anomalies (far from the cluster) need very FEW cuts.
        The algorithm scores each point by how many cuts it took to isolate it.
        Low score = anomaly.

        WHY THIS IS BETTER THAN 2-SIGMA:
        It uses MULTIPLE features at once — not just raw flow, but also
        the 7-day average, standard deviation, and ratio. So it can catch
        anomalies that look normal on one dimension but weird on another.

        contamination = the fraction of data we expect to be anomalous.
        0.05 means "I expect about 5% of days to be unusual."
        """
       
        features = ["flow_cfs", "flow_7day_avg", "flow_7day_std", "flow_ratio"]
        X = self.df[features].fillna(0)   # fill any NaN with 0 for the model

       
        self.model = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100,    # number of isolation trees — more = more stable
        )

        
        predictions = self.model.fit_predict(X)

        # Also get the raw anomaly score (lower = more anomalous)
        scores = self.model.score_samples(X)

        
        self.df["anomaly_flag"]  = predictions          # -1 or 1
        self.df["anomaly_score"] = scores.round(4)

        
        anomaly_mask   = self.df["anomaly_flag"] == -1
        self.anomalies = self.df[anomaly_mask].copy()
        self.anomalies["method"] = "isolation_forest"

        logger.info(f"Isolation Forest: {len(self.anomalies)} anomalies detected")
        logger.info(f"Anomaly dates: {[str(d.date()) for d in self.anomalies.index]}")

        return self.anomalies


    def compare_methods(self) -> None:
        """
        Compares the statistical (2-sigma) and ML (Isolation Forest) methods.

        This shows something important: the two methods don't always agree.
        The ML model might catch a day that looked statistically normal but
        had a suspicious PATTERN — e.g. unusually high flow for 6 consecutive
        days even if no single day broke the 2-sigma threshold.
        """
        statistical = self.detect_anomalies_statistical()
        ml          = self.detect_anomalies_ml()

        stat_dates = set(statistical.index)
        ml_dates   = set(ml.index)

        both   = stat_dates & ml_dates      
        only_stat = stat_dates - ml_dates   
        only_ml   = ml_dates - stat_dates   

        print("\n══ METHOD COMPARISON ═══════════════════════════════")
        print(f"  Statistical (2σ) anomalies : {len(stat_dates)}")
        print(f"  Isolation Forest anomalies : {len(ml_dates)}")
        print(f"  Agreed on (both flagged)   : {len(both)}")
        print(f"  Only statistical flagged   : {len(only_stat)}")
        print(f"  Only ML flagged            : {len(only_ml)}")

        if both:
            print(f"\n  ✓ Both methods agree on: {sorted([d.date() for d in both])}")
        if only_ml:
            print(f"\n  ★ ML-only detections (subtle patterns):")
            for date in sorted(only_ml):
                row = self.df.loc[date]
                print(f"    {date.date()}  flow={row['flow_cfs']:.2f}  ratio={row['flow_ratio']:.2f}  score={row['anomaly_score']:.4f}")
        print("════════════════════════════════════════════════════\n")


    def generate_report(self) -> None:
        """
        Writes a structured text report to a file.

        This demonstrates:
          1. Calling other methods from within a method (self.compute_statistics())
          2. Writing multi-line formatted text to a file
          3. Using f-strings to embed variables in text cleanly

        The report is something you can show an employer or include in a README.
        """
        stats = self.compute_statistics()

        # Make sure anomalies are detected before generating the report
        if self.anomalies.empty:
            self.detect_anomalies_ml()

        lines = [
            "═" * 60,
            "  WATER QUALITY MONITORING REPORT",
            f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "═" * 60,
            "",
            "STATION INFORMATION",
            f"  Name      : {stats['station']}",
            f"  Period    : {stats['date_start']}  →  {stats['date_end']}",
            f"  Days      : {stats['days_total']}",
            "",
            "FLOW STATISTICS",
            f"  Mean flow   : {stats['mean_flow']} cfs",
            f"  Median flow : {stats['median_flow']} cfs",
            f"  Std dev     : {stats['std_flow']} cfs",
            f"  Peak flow   : {stats['max_flow']} cfs  ({stats['max_flow_date']})",
            f"  Lowest flow : {stats['min_flow']} cfs  ({stats['min_flow_date']})",
            f"  Low-flow days (<0.5 cfs): {stats['pct_low_flow']}% of record",
            "",
            "ANOMALY DETECTION (Isolation Forest)",
            f"  Total anomalous days: {len(self.anomalies)}",
            "",
            "  Date          Flow (cfs)   7-day avg    Ratio    Score",
            "  " + "-" * 56,
        ]

        # List comprehension — builds a formatted line for each anomaly row
        # This is a one-line for loop that produces a list of strings
        anomaly_lines = [
            f"  {date.date()}    {row['flow_cfs']:>8.2f}     {row['flow_7day_avg']:>6.2f}     {row['flow_ratio']:>5.2f}    {row['anomaly_score']:>6.4f}"
            for date, row in self.anomalies.sort_values("flow_cfs", ascending=False).iterrows()
        ]

        lines.extend(anomaly_lines)
        lines.extend([
            "",
            "INTERPRETATION",
            "  Days with anomaly_score near -0.5 or lower are most suspicious.",
            "  High flow_ratio (>3.0) means the day was 3x above recent average.",
            "  Cross-reference spike dates with local precipitation records",
            "  to confirm storm events vs potential sensor or data issues.",
            "",
            "═" * 60,
        ])

        report_text = "\n".join(lines)

        
        print(report_text)

        
        with open(self.report_path, "w", encoding="utf-8") as f:
            f.write(report_text)

        logger.info(f"Report saved → {self.report_path}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # This is the payoff of OOP — three lines to run the whole analysis
    monitor = WaterQualityMonitor("streamflow_data.csv")
    monitor.compare_methods()
    monitor.generate_report()
