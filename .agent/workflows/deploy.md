---
description: Deploy backtest results to GitHub Pages
---

# Deploy Workflow

## Prerequisites

- GitHub Pages must be enabled in repository settings
- Source: GitHub Actions

## Steps

1. Commit latest results:

```bash
git add *.html
git commit -m "Update backtest results"
```

2. Push to trigger GitHub Actions:

```bash
git push origin main
```

3. Monitor deployment:

- Go to GitHub → Actions tab
- Check `daily_backtest.yml` workflow status

4. View live results:

- URL: `https://<username>.github.io/MYC_GDCA/`

## Manual GitHub Actions Trigger

If you need to run the workflow manually:

1. Go to Actions → daily_backtest
2. Click "Run workflow"
