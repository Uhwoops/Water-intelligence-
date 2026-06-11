"""
water_fetcher.py  —  Water Quality Intelligence Pipeline
Fetches streamflow data from the USGS Water Services API and saves it to a CSV.
"""

import requests
import csv
from datetime import datetime, timedelta


STATION_ID     = "02146470"
DAYS_BACK      = 120
PARAMETER_CODE = "00060"
OUTPUT_FILE    = "streamflow_data.csv"


def build_url(station_id: str, days_back: int, param_code: str) -> str:
    """Builds the USGS API URL for a given station, date range, and parameter."""
    end_date   = datetime.today() - timedelta(days=3)
    start_date = end_date - timedelta(days=days_back)

    url = (
        f"https://waterservices.usgs.gov/nwis/dv/"
        f"?format=json"
        f"&sites={station_id}"
        f"&startDT={start_date.strftime('%Y-%m-%d')}"
        f"&endDT={end_date.strftime('%Y-%m-%d')}"
        f"&parameterCd={param_code}"
        f"&siteStatus=all"
    )
    return url


def fetch_data(url: str) -> dict | None:
    """Makes the HTTP GET request and returns the parsed JSON response."""
    try:
        print(f"Fetching: {url}\n")
        response = requests.get(url, timeout=10)
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
    Extracts daily mean flow readings from the USGS JSON response.

    USGS returns multiple time series per request (Maximum, Minimum, Mean).
    This function finds the Mean series (statistic code 00003) and extracts
    its values. Falls back to the longest available series if Mean is absent.
    """
    readings = []

    try:
        time_series = raw_data["value"]["timeSeries"]

        if not time_series:
            print("No data returned for this station / date range.")
            return readings

        chosen_series = None
        chosen_values = []

        for series in time_series:
            options   = series["variable"]["options"]["option"]
            stat_code = None

            for opt in options:
                if opt.get("name") == "Statistic":
                    stat_code = opt.get("optionCode")

            value_list = series["values"][0]["value"]
            print(f"  Found series: statCode={stat_code}, rows={len(value_list)}")

            if stat_code == "00003" and len(value_list) > 0:
                chosen_series = series
                chosen_values = value_list
                break

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
            date_str  = entry["dateTime"].split("T")[0]
            raw_value = entry["value"]

            flow_cfs = None if raw_value in ("-999999", "") else float(raw_value)

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
    """Writes the readings list to a CSV file."""
    if not readings:
        print("Nothing to save.")
        return

    fieldnames = ["date", "station_id", "site_name", "flow_cfs"]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(readings)

    print(f"Saved {len(readings)} rows to {filepath}")


def summarize(readings: list[dict]) -> None:
    """Prints a summary of the fetched readings."""
    flows = [r["flow_cfs"] for r in readings if r["flow_cfs"] is not None]

    if not flows:
        print("No valid flow readings to summarize.")
        return

    print("── Summary ──────────────────────────────")
    print(f"  Days fetched  : {len(readings)}")
    print(f"  Valid readings: {len(flows)}")
    print(f"  Average flow  : {sum(flows)/len(flows):,.1f} cfs")
    print(f"  Peak flow     : {max(flows):,.1f} cfs")
    print(f"  Lowest flow   : {min(flows):,.1f} cfs")
    print("─────────────────────────────────────────")


if __name__ == "__main__":
    url      = build_url(STATION_ID, DAYS_BACK, PARAMETER_CODE)
    raw_data = fetch_data(url)

    if raw_data:
        readings = parse_readings(raw_data)
        summarize(readings)
        save_to_csv(readings, OUTPUT_FILE)
