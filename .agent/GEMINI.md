# MYC_GDCA Project Rules

## Project Context

This is a cryptocurrency backtesting project implementing GDCA (Grospector DCA) strategy using Nautilus Trader and CCXT.

## Technology Stack

- **Python 3.11+** with Poetry for dependency management
- **Nautilus Trader** for backtesting engine
- **CCXT** for exchange data
- **HTML/JavaScript** (Lightweight Charts) for visualization
- **GitHub Actions** for daily automation

## Conventions

- Strategy files → `strategies/`
- Configuration → `config/settings.py`
- Visualization → `templates/`
- Scripts → `scripts/`
- Results → `*_result.html` files

## Code Standards

- Use type hints for all functions
- Follow PEP 8 style guide
- Docstrings for public functions
- Keep strategies modular and testable

## UI/UX Standards

- **All HTML pages MUST be mobile-responsive**
- Extend `templates/base.html` for consistent styling
- Include responsive breakpoints: 1024px, 768px, 480px
- Hamburger menu required for mobile navigation
- Test all pages at 375px width before deployment
- See `.agent/skills/project-helpers/SKILL.md` for detailed UI guidelines

## Common Commands

```bash
# Install dependencies
poetry install

# Download latest data
poetry run python scripts/download_data.py

# Run backtest
poetry run python scripts/run_backtest.py
```

## Zone Logic Reference

| Zone        | Condition          | Action                   |
| ----------- | ------------------ | ------------------------ |
| Short       | Overvalued         | Potential short entries  |
| Strong Sell | High valuation     | Aggressive profit taking |
| Sell        | Moderate valuation | Standard profit taking   |
| Buy         | Fair value         | Standard DCA             |
| Strong Buy  | Undervalued        | Aggressive accumulation  |
| Long        | Deep undervalued   | Maximum accumulation     |
