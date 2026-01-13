"""
HTML Report Generator
=====================

This module generates HTML reports using Jinja2 templates.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import json
import sys

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from templates.renderer import render_template, save_html


def generate_strategy_report(
    output_path: Path,
    strategy_name: str,
    strategy_color: str,
    metrics: Dict[str, Any],
    chart_data: Dict[str, Any],
    is_gdca: bool = True,
) -> None:
    """
    Generate an individual strategy report HTML file.
    
    Args:
        output_path: Path to save the HTML file.
        strategy_name: Name of the strategy (e.g., "GDCA Strategy").
        strategy_color: Primary color for the strategy.
        metrics: Dictionary of performance metrics.
        chart_data: Dictionary of chart data series (JSON strings).
        is_gdca: Whether this is the GDCA strategy. Defaults to True.
    """
    # Prepare template context
    context = {
        "strategy_name": strategy_name,
        "strategy_color": strategy_color,
        "is_gdca": is_gdca,
        "metrics": metrics,
        **chart_data,  # Spread chart data into context
    }
    
    # Render and save
    html = render_template("strategy.html", **context)
    save_html(html, output_path)


def generate_comparison_report(
    output_path: Path,
    gdca_metrics: Dict[str, Any],
    dca_metrics: Dict[str, Any],
    chart_data: Dict[str, Any],
) -> None:
    """
    Generate a comparison report HTML file.
    
    Args:
        output_path: Path to save the HTML file.
        gdca_metrics: GDCA strategy metrics.
        dca_metrics: Standard DCA strategy metrics.
        chart_data: Dictionary of chart data series.
    """
    # Create metrics objects for template
    class MetricsWrapper:
        def __init__(self, data: dict):
            for key, value in data.items():
                setattr(self, key, value)
    
    gdca_wrapper = MetricsWrapper(gdca_metrics)
    dca_wrapper = MetricsWrapper(dca_metrics)
    
    # Determine winner
    gdca_roi = gdca_metrics.get('roi', 0)
    dca_roi = dca_metrics.get('std_dca_roi', 0)
    gdca_wins = gdca_roi > dca_roi
    
    # Prepare template context
    context = {
        "gdca_metrics": gdca_wrapper,
        "dca_metrics": dca_wrapper,
        "gdca_wins": gdca_wins,
        **chart_data,
    }
    
    # Render and save
    html = render_template("comparison.html", **context)
    save_html(html, output_path)


def prepare_chart_data_json(
    ohlc_data: list,
    equity_data: list,
    bnh_data: list,
    cash_data: list = None,
    holdings_data: list = None,
    markers_data: list = None,
    ema12_data: list = None,
    ema26_data: list = None,
    ribbons_data: list = None,
    zone_data: dict = None,
) -> Dict[str, str]:
    """
    Prepare chart data as JSON strings for template rendering.
    
    Args:
        ohlc_data: OHLC candlestick data.
        equity_data: Equity line data.
        bnh_data: Buy-and-hold/invested capital data.
        cash_data: Cash balance data. Optional.
        holdings_data: BTC holdings data. Optional.
        markers_data: Trade markers data. Optional.
        ema12_data: EMA-12 indicator data. Optional.
        ema26_data: EMA-26 indicator data. Optional.
        ribbons_data: Ribbon lines data. Optional.
        zone_data: Zone boundary data. Optional.
        
    Returns:
        Dictionary of JSON-encoded data strings.
    """
    result = {
        "json_ohlc": json.dumps(ohlc_data),
        "json_equity": json.dumps(equity_data),
        "json_bnh": json.dumps(bnh_data),
    }
    
    if cash_data is not None:
        result["json_cash"] = json.dumps(cash_data)
    
    if holdings_data is not None:
        result["json_holdings"] = json.dumps(holdings_data)
    
    if markers_data is not None:
        result["json_markers"] = json.dumps(markers_data)
    
    if ema12_data is not None:
        result["json_ema12"] = json.dumps(ema12_data)
    
    if ema26_data is not None:
        result["json_ema26"] = json.dumps(ema26_data)
    
    if ribbons_data is not None:
        result["json_ribbons"] = json.dumps(ribbons_data)
    
    if zone_data is not None:
        for zone_name, data in zone_data.items():
            result[f"json_{zone_name}"] = json.dumps(data)
    
    return result


def prepare_comparison_chart_data(
    gdca_equity: list,
    dca_equity: list,
    bnh_data: list,
) -> Dict[str, str]:
    """
    Prepare chart data for comparison report.
    
    Args:
        gdca_equity: GDCA equity curve data.
        dca_equity: Standard DCA equity curve data.
        bnh_data: Invested capital data.
        
    Returns:
        Dictionary of JSON-encoded data strings.
    """
    return {
        "json_equity_gdca": json.dumps(gdca_equity),
        "json_equity_dca": json.dumps(dca_equity),
        "json_bnh": json.dumps(bnh_data),
    }
