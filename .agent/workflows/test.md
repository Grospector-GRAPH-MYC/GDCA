---
description: Run the test suite for the project
---

# Test Workflow

## Steps

1. Install dependencies:
   // turbo

```bash
poetry install
```

2. Run tests with pytest:
   // turbo

```bash
poetry run pytest -v
```

3. Run tests with coverage:

```bash
poetry run pytest --cov=strategies --cov=config --cov-report=html
```

4. View coverage report (if generated):

```bash
start htmlcov/index.html
```
