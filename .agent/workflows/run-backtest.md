---
description: Run a backtest on the GDCA strategy with latest data
---

# Run Backtest Workflow

## Steps

1. Install dependencies (if needed):
   // turbo

```bash
poetry install
```

2. Download latest market data:
   // turbo

```bash
poetry run python scripts/download_data.py
```

3. Run the backtest:
   // turbo

```bash
poetry run python scripts/run_backtest.py
```

4. View results:

```bash
start gdca_result.html
```

## Notes

- Results are saved to `gdca_result.html`, `dca_result.html`, and `comparison_result.html`
- Check `error.log` if issues occur
