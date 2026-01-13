"""
Visualization Module
====================

This module provides utilities for preparing chart data and generating HTML reports.

Submodules:
    - chart_data: Data preparation for Lightweight Charts
    - html_generator: HTML report generation using Jinja2 templates
"""

from visualization.chart_data import (
    prepare_ohlc_data,
    prepare_equity_data,
    prepare_marker_data,
    prepare_indicator_data,
    prepare_ribbon_data,
)
from visualization.html_generator import (
    generate_strategy_report,
    generate_comparison_report,
)

__all__ = [
    "prepare_ohlc_data",
    "prepare_equity_data",
    "prepare_marker_data",
    "prepare_indicator_data",
    "prepare_ribbon_data",
    "generate_strategy_report",
    "generate_comparison_report",
]
