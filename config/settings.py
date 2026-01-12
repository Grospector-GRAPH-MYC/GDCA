import os
from pathlib import Path
from dotenv import load_dotenv

class Settings:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Settings, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        # Load .env
        load_dotenv()
        
    def _get(self, key: str, section: str = "strategy", default=None, type_func=str):
        # 1. Env Var (Upper Case)
        env_key = key.upper()
        env_val = os.getenv(env_key)
        if env_val is not None:
            try:
                return type_func(env_val)
            except ValueError:
                print(f"Warning: Could not cast env var {env_key}={env_val} to {type_func}")
        
        # 2. Default
        return default

    # --- Strategy Config ---
    @property
    def SYMBOL(self) -> str:
        return self._get("symbol", default="BTC/USD")

    @property
    def LOT_SIZE(self) -> str:
        return self._get("lot_size", default="0.001")

    @property
    def STEP_DROP_PCT(self) -> str:
        return self._get("step_drop_pct", default="0.01")
    
    @property
    def TAKE_PROFIT_PCT(self) -> str:
        return self._get("take_profit_pct", default="0.05")

    @property
    def MAX_POSITIONS(self) -> int:
        return self._get("max_positions", default=10, type_func=int)
    
    @property
    def DCA_AMOUNT(self) -> float:
        return self._get("dca_amount", default=1000.0, type_func=float)

    @property
    def TIMEFRAME(self) -> str:
        # Default fallback for strategy execution if not in config
        return self._get("timeframe", default="1d")

    # --- Zone Config ---
    @property
    def START_SHORT_ZONE(self) -> int: return self._get("start_short_zone", default=100, type_func=int)
    @property
    def END_SHORT_ZONE(self) -> int: return self._get("end_short_zone", default=0, type_func=int)
    
    @property
    def START_SELL_ZONE(self) -> int: return self._get("start_sell_zone", default=100, type_func=int)
    @property
    def END_SELL_ZONE(self) -> int: return self._get("end_sell_zone", default=0, type_func=int)
    
    @property
    def START_NORMAL_ZONE(self) -> int: return self._get("start_normal_zone", default=0, type_func=int)
    @property
    def END_NORMAL_ZONE(self) -> int: return self._get("end_normal_zone", default=100, type_func=int)
    
    @property
    def START_BUY_ZONE(self) -> int: return self._get("start_buy_zone", default=100, type_func=int)
    @property
    def END_BUY_ZONE(self) -> int: return self._get("end_buy_zone", default=200, type_func=int)
    
    @property
    def START_LONG_ZONE(self) -> int: return self._get("start_long_zone", default=0, type_func=int)
    @property
    def END_LONG_ZONE(self) -> int: return self._get("end_long_zone", default=100, type_func=int)

    # --- Nautilus / General Config ---
    @property
    def VENUE(self) -> str:
        # Check 'strategy' section first (common mistake), then 'nautilus' section
        val = self._get("venue", section="nautilus", default=None)
        if val is None:
             val = self._get("venue", section="strategy", default="BITSTAMP")
        return val

    # --- Data Download Config ---
    @property
    def DATA_SYMBOL(self) -> str:
        return self._get("data_symbol", section="data", default="BTC/USD")
    
    @property
    def DATA_VENUE(self) -> str:
        return self._get("data_venue", section="data", default="BITSTAMP")
        
    @property
    def DATA_TIMEFRAME(self) -> str:
        return self._get("data_timeframe", section="data", default="1d")
        
    @property
    def DATA_START_DATE(self) -> str:
        return self._get("data_start_date", section="data", default="2011-01-01")
    
    @property
    def DATA_TYPE(self) -> str:
        return self._get("data_type", section="data", default="bar")

    # --- Backtest Config ---
    @property
    def BACKTEST_START_DATE(self) -> str:
        return self._get("backtest_start_date", section="backtest", default=None)

    @property
    def BACKTEST_END_DATE(self) -> str:
        return self._get("backtest_end_date", section="backtest", default=None)

# Singleton Instance
settings = Settings()
