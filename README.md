\# Water Quality Intelligence Pipeline

> Real-time streamflow monitoring and anomaly detection for Little Hope Creek, Charlotte, NC



\## Overview

A full data pipeline that ingests live USGS streamflow data, performs time-series analysis, detects anomalies using machine learning, and delivers results through an interactive dashboard.



\## Charts

!\[Daily Flow](charts/chart1\_daily\_flow.png)

!\[Spike Detection](charts/chart3\_spike\_detection.png)



\## Pipeline

| Script | Purpose |

|--------|---------|

| `water\_fetcher.py` | Fetches live USGS streamflow data via REST API |

| `water\_analyzer.py` | Pandas time-series analysis and statistics |

| `water\_visualizer.py` | Matplotlib chart generation |

| `water\_monitor.py` | ML anomaly detection with Isolation Forest |

| `water\_dashboard.html` | Interactive HTML5 dashboard |



\## Key Findings

\- 121 days of data from USGS Station 02146470

\- Peak flow: 29.4 cfs on Feb 15, 2026 (16x average)

\- 6 anomalous storm events detected by ML model

\- ML caught 1 event missed by statistical methods

\- 24.8% of days below 0.5 cfs (near-dry conditions)



\## Tech Stack

Python · pandas · scikit-learn · matplotlib · requests · HTML5 · CSS3 · Chart.js



\## Setup

```bash

pip install requests pandas matplotlib scikit-learn

python water\_fetcher.py

python water\_analyzer.py

python water\_visualizer.py

python water\_monitor.py

```

Open `water\_dashboard.html` in your browser to view the dashboard.



\## Data Source

USGS Water Services API — Station 02146470

Little Hope Creek at Seneca Place, Charlotte, NC

