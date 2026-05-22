"""
water_fetcher.py  —  Week 1 of the Water Quality Intelligence Project
Fetches streamflow data from the USGS Water Services API and saves it to a CSV.
"""

import requests   
import csv        
import json       
from datetime import datetime, timedelta  


# ── Configuration ────────────────────────────────────────────────────────────
STATION_ID = "02146470"

# How many days back to fetch
DAYS_BACK = 120

# The parameter code for streamflow (discharge) in cubic feet per second
PARAMETER_CODE = "00060"

OUTPUT_FILE = "streamflow_data.csv"
# ─────────────────────────────────────────────────────────────────────────────


def build_url(station_id: str, days_back: int, param_code: str) -> str:
    """
    Builds the USGS API URL with the right query parameters.

    A URL has two parts:
      - The base path:  https://waterservices.usgs.gov/nwis/dv/
      - Query params:   ?site=...&startDT=...&endDT=...&format=json
    """
    end_date   = datetime.today() - timedelta(days=3)   # USGS daily values lag ~1-3 days
    start_date = end_date - timedelta(days=days_back)

    # strftime() formats a datetime object into a string
    # "%Y-%m-%d" means: 4-digit year, month, day  →  "2025-04-18"
    start_str = start_date.strftime("%Y-%m-%d")
    end_str   = end_date.strftime("%Y-%m-%d")

    url = (
        f"https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json"
        f"&sites={station_id}"
        f"&startDT={start_str}"
        f"&endDT={end_str}"
        f"&parameterCd={param_code}"
        f"&siteStatus=all"
    )
    return url


def fetch_data(url: str) -> dict | None:
   
    try:
        print(f"Fetching: {url}\n")
        response = requests.get(url, timeout=10)

        # raise_for_status() throws an error if the server returned 4xx or 5xx
        response.raise_for_status()

        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Network error: {e}")
        return None
    except ValueError as e:
        print(f"Could not parse JSON: {e}")
        return None


def parse_readings(raw_data: dict) -> list[dict]:
    """
    Digs into the nested JSON and extracts daily flow readings.

    WHY THIS VERSION IS MORE ROBUST:
    Your station returns MULTIPLE time series in one response — one for
    the Mean statistic, one for Maximum, one for Minimum. The old code
    blindly took [0] which hit a series with 0 approved values.

    This version loops every series and picks the one whose statistic
    option is "00003" (Mean) and actually has data in it.

    Statistic codes:
      00001 = Maximum
      00002 = Minimum
      00003 = Mean  ← the one we want
    """
    readings = []

    try:
        time_series = raw_data["value"]["timeSeries"]

        if not time_series:
            print("No data returned for this station / date range.")
            return readings

       
        chosen_series  = None
        chosen_values  = []

        for series in time_series:
            # Each series has an "options" list describing its statistic type
            options     = series["variable"]["options"]["option"]
            stat_code   = None

            for opt in options:
                if opt.get("name") == "Statistic":
                    stat_code = opt.get("optionCode")

           
            value_list = series["values"][0]["value"]

            print(f"  Found series: statCode={stat_code}, rows={len(value_list)}")

         
            if stat_code == "00003" and len(value_list) > 0:
                chosen_series = series
                chosen_values = value_list
                break   # stop as soon as we find a good one

        if not chosen_series:
            
            print("\n  No Mean series found — falling back to longest series with data.")
            for series in time_series:
                value_list = series["values"][0]["value"]
                if len(value_list) > len(chosen_values):
                    chosen_series = series
                    chosen_values = value_list

        if not chosen_series:
            print("Could not find any series with data.")
            return readings

        site_name  = chosen_series["sourceInfo"]["siteName"]
        param_name = chosen_series["variable"]["variableName"]

        print(f"\nStation  : {site_name}")
        print(f"Parameter: {param_name}")
        print(f"Readings : {len(chosen_values)} days\n")

        for entry in chosen_values:
            raw_dt    = entry["dateTime"]
            raw_value = entry["value"]

            date_str = raw_dt.split("T")[0]   

          
            if raw_value in ("-999999", ""):
                flow_cfs = None
            else:
                flow_cfs = float(raw_value)

            readings.append({
                "date":       date_str,
                "flow_cfs":   flow_cfs,
                "site_name":  site_name,
                "station_id": STATION_ID,
            })

    except (KeyError, IndexError, TypeError) as e:
        print(f"Error parsing data structure: {e}")

    return readings


def save_to_csv(readings: list[dict], filepath: str) -> None:
    """
    Writes the list of reading dictionaries to a CSV file.

    csv.DictWriter takes the fieldnames (column headers) and writes
    each dict as one row — matching keys to columns automatically.

    'with open(...)' is a context manager. It guarantees the file
    is properly closed after the block exits, even if an error occurs.
    """
    if not readings:
        print("Nothing to save.")
        return

    fieldnames = ["date", "station_id", "site_name", "flow_cfs"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()          # writes the column name row
        writer.writerows(readings)    # writes all data rows at once

    print(f"Saved {len(readings)} rows → {filepath}")


def summarize(readings: list[dict]) -> None:
    """
    Prints a quick text summary so you can sanity-check the data
    before moving on to pandas in Week 2.

    This uses a list comprehension — [x for x in list if condition] —
    to filter out None values before doing math.
    """
    flows = [r["flow_cfs"] for r in readings if r["flow_cfs"] is not None]

    if not flows:
        print("No valid flow readings to summarize.")
        return

    avg_flow = sum(flows) / len(flows)
    max_flow = max(flows)
    min_flow = min(flows)

    print("── Summary ──────────────────────────────")
    print(f"  Days fetched : {len(readings)}")
    print(f"  Valid readings: {len(flows)}")
    print(f"  Average flow : {avg_flow:,.1f} cfs")
    print(f"  Peak flow    : {max_flow:,.1f} cfs")
    print(f"  Lowest flow  : {min_flow:,.1f} cfs")
    print("─────────────────────────────────────────")


# ── Main entry point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    url      = build_url(STATION_ID, DAYS_BACK, PARAMETER_CODE)
    raw_data = fetch_data(url)

    if raw_data:
        readings = parse_readings(raw_data)
        summarize(readings)
        save_to_csv(readings, OUTPUT_FILE)
