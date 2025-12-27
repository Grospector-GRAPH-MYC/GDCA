# GDCA Strategy (Grospector DCA)

An automated Dollar-Cost Averaging (DCA) trading system built with **Nautilus Trader** and **CCXT**. This project implements a dynamic DCA strategy that adjusts investment position size based on market "Zones" derived from Moving Averages and CDC Action Zone indicators.

## 🚀 Features

-   **Dynamic DCA**: Adjusts buy amounts based on price levels relative to long-term Moving Average ribbons.
-   **Zone-Based Logic**: 6 Distinct Zones (Short, Strong Sell, Sell, Buy, Strong Buy, Long) to optimize entry and exit.
-   **CDC Action Zone**: Integrated trend confirmation using Fast/Slow EMAs.
-   **Zero-Cost Automation**: configured to run daily via **GitHub Actions** for free.
-   **Visual Reporting**: Generates an interactive `backtest_result.html` using Lightweight Charts (TradingView style).

## 📊 Strategy Logic

The strategy defines channels using a **Base MA** (1460 days / 4 years) and an **MA of MA** (365 days). Multipliers are applied to these baselines to create price bands:

1.  **Short Zone**: Excessive overvaluation. (Potential Short entries).
2.  **Strong Sell**: High valuation. Aggressive profit taking.
3.  **Sell (Normal)**: Moderate valuation. Standard profit taking.
4.  **Buy**: Fair value. Standard DCA accumulation.
5.  **Strong Buy**: Undervalued. Aggressive accumulation.
6.  **Long**: Deep undervaluation. Maximum accumulation.

### Execution
-   **Accumulation**: A daily "Savings" amount (`DCA_AMOUNT`) is added to a virtual reserve.
-   **Investment**: The strategy calculates a % of capital to deploy based on the current Zone depth.
-   **Aggressive Mode**: If the signals are strong (>100%), it dips into the accumulated reserve for larger buys.
-   **Exit**: Positions are closed when price hits the Sell Zone and the Trend turns Bearish.

## 🛠️ Setup & Usage

### Prerequisites
-   Python 3.11+
-   [Poetry](https://python-poetry.org/)

### Installation

```bash
# Install dependencies
poetry install
```

### Running Locally

1.  **Download Data**:
    ```bash
    poetry run python scripts/download_data.py
    ```

2.  **Run Backtest**:
    ```bash
    poetry run python scripts/run_backtest.py
    ```

3.  **View Results**:
    Open `backtest_result.html` in your browser.

## 🤖 Daily Automation (GitHub Actions)

This repository includes a workflow `.github/workflows/daily_backtest.yml` that runs automatically:

1.  **Schedule**: Runs every day at 00:00 UTC.
2.  **Action**: Fetches latest daily data -> Runs Strategy -> Generates Report.
3.  **Deployment**: Deploys the interactive chart to **GitHub Pages**.

### Enabling GitHub Pages
To view your daily report:
1.  Go to **Settings** -> **Pages** in your repository.
2.  Under **Build and deployment**, select **GitHub Actions** as the source.
3.  After the next run, your site will be live at `https://<user>.github.io/<repo>/`.

## ⚙️ Configuration

You can tweak the strategy via Environment Variables or `.env` file:

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DCA_AMOUNT` | `1000.0` | Daily USD amount added to generic savings |
| `START_BUY_ZONE` | `100` | Starting % for Buy Zone |
| `manual_buy_multi` | `1.0` | Multiplier for Fair Value calculation |

*(See `strategies/gdca_strategy.py` for full config breakdown)*
