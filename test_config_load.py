
import sys
from pathlib import Path
import os

# Set Env Var for test
os.environ["LOT_SIZE"] = "0.888"
os.environ["DATA_SYMBOL"] = "ETH/USD"

from config.settings import settings

print(f"LOT_SIZE (Env Override): {settings.LOT_SIZE}")
print(f"SYMBOL (Default/TOML): {settings.SYMBOL}")
print(f"DATA_SYMBOL (Env Override): {settings.DATA_SYMBOL}")

# Ensure singleton behavior (though basic import usually guarantees this in Python)
from config.settings import Settings
s2 = Settings()
print(f"Is singleton? {settings is s2}")
