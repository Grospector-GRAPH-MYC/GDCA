"""
Backtest Module
===============

This module provides utilities for running backtests with Nautilus Trader.

Submodules:
    - engine: BacktestEngine setup and configuration
    - data_processor: Data loading and preparation
    - metrics: Performance metrics calculation
    - simulator: DCA simulation logic
"""

from backtest.engine import create_engine, create_instrument, add_venue
from backtest.data_processor import load_bars, prepare_bar_dataframe
from backtest.metrics import calculate_metrics, calculate_drawdown
from backtest.simulator import simulate_standard_dca

__all__ = [
    "create_engine",
    "create_instrument", 
    "add_venue",
    "load_bars",
    "prepare_bar_dataframe",
    "calculate_metrics",
    "calculate_drawdown",
    "simulate_standard_dca",
]
