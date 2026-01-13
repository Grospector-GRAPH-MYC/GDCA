"""
HTML Report Generator Module

This module provides functions to generate HTML reports from backtest data
using Jinja2 templates. It's designed to be called from run_backtest.py
as an alternative to the embedded HTML generators.
"""

from pathlib import Path
from templates.renderer import render_strategy, render_comparison, render_index, save_html


def generate_all_reports(
    # GDCA metrics
    total_invested: float,
    total_equity: float,
    profit_usd: float,
    roi_pct: float,
    max_dd_pct: float,
    total_btc: float,
    total_trades: int,
    win_rate_pct: float,
    # Standard DCA metrics
    std_dca_invested: float,
    std_dca_current_value: float,
    std_dca_net_profit: float,
    std_dca_roi: float,
    std_dca_max_dd_pct: float,
    std_dca_btc: float,
    std_dca_trades: int,
    # Chart data (JSON strings)
    json_ohlc: str,
    json_equity: str,
    json_std_dca_equity: str,
    json_bnh: str,
    json_cash: str,
    json_holdings: str,
    json_markers: str,
    json_dca_markers: str,
    json_dca_holdings: str,
    json_dca_cash: str,
    json_ema12: str,
    json_ema26: str,
    json_ribbons: str,
    json_ribbons_past: str,
    json_future: str,
    json_past: str,
    json_ma_short: str,
    json_ma_strong_sell: str,
    json_ma_sell: str,
    json_ma_buy: str,
    json_ma_strong_buy: str,
    json_ma_long: str,
    # Backtest date range
    backtest_start: str = None,
    backtest_end: str = None,
    # Output path
    output_dir: str = "."
) -> None:
    """
    Generate all HTML reports using Jinja2 templates.
    
    Args:
        All metric values and JSON data strings from backtest results.
        backtest_start: Start date of backtest period (YYYY-MM-DD)
        backtest_end: End date of backtest period (YYYY-MM-DD)
        output_dir: Directory to save HTML files (default: current directory)
    """
    output_path = Path(output_dir)
    
    # GDCA Metrics dict
    gdca_metrics = {
        'total_invested': total_invested,
        'total_equity': total_equity,
        'net_profit': profit_usd,
        'roi': roi_pct,
        'max_drawdown': max_dd_pct,
        'total_btc': total_btc,
        'total_trades': total_trades,
        'win_rate': win_rate_pct
    }
    
    # DCA Metrics dict
    dca_metrics = {
        'std_dca_invested': std_dca_invested,
        'std_dca_equity': std_dca_current_value,
        'std_dca_profit': std_dca_net_profit,
        'std_dca_roi': std_dca_roi,
        'std_dca_max_drawdown': std_dca_max_dd_pct,
        'std_dca_btc': std_dca_btc,
        'std_dca_trades': std_dca_trades
    }
    
    # Generate GDCA HTML
    gdca_html = render_strategy(
        strategy_name="GDCA Strategy",
        strategy_color="#3b82f6",
        is_gdca=True,
        invested=total_invested,
        value=total_equity,
        profit=profit_usd,
        roi=roi_pct,
        max_dd=max_dd_pct,
        btc_held=total_btc,
        trades=total_trades,
        json_ohlc=json_ohlc,
        json_equity=json_equity,
        json_bnh=json_bnh,
        json_cash=json_cash,
        json_holdings=json_holdings,
        json_markers=json_markers,
        json_ema12=json_ema12,
        json_ema26=json_ema26,
        json_ribbons=json_ribbons,
        json_ribbons_past=json_ribbons_past,
        json_future=json_future,
        json_past=json_past,
        json_ma_short=json_ma_short,
        json_ma_strong_sell=json_ma_strong_sell,
        json_ma_sell=json_ma_sell,
        json_ma_buy=json_ma_buy,
        json_ma_strong_buy=json_ma_strong_buy,
        json_ma_long=json_ma_long,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )
    save_html(gdca_html, str(output_path / 'gdca_result.html'))
    print("Generated: gdca_result.html")
    
    # Generate Standard DCA HTML
    dca_html = render_strategy(
        strategy_name="Standard DCA",
        strategy_color="#8b5cf6",
        is_gdca=False,
        invested=std_dca_invested,
        value=std_dca_current_value,
        profit=std_dca_net_profit,
        roi=std_dca_roi,
        max_dd=std_dca_max_dd_pct,
        btc_held=std_dca_btc,
        trades=std_dca_trades,
        json_ohlc=json_ohlc,
        json_equity=json_std_dca_equity,
        json_bnh=json_bnh,
        json_cash=json_dca_cash,
        json_holdings=json_dca_holdings,
        json_markers=json_dca_markers,
        json_ema12=json_ema12,
        json_ema26=json_ema26,
        json_ribbons=json_ribbons,
        json_ribbons_past=json_ribbons_past,
        json_future=json_future,
        json_past=json_past,
        json_ma_short=json_ma_short,
        json_ma_strong_sell=json_ma_strong_sell,
        json_ma_sell=json_ma_sell,
        json_ma_buy=json_ma_buy,
        json_ma_strong_buy=json_ma_strong_buy,
        json_ma_long=json_ma_long,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )
    save_html(dca_html, str(output_path / 'dca_result.html'))
    print("Generated: dca_result.html")
    
    # Generate Comparison HTML
    comparison_html = render_comparison(
        gdca_metrics=gdca_metrics,
        dca_metrics=dca_metrics,
        json_equity_gdca=json_equity,
        json_equity_dca=json_std_dca_equity,
        json_bnh=json_bnh,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )
    save_html(comparison_html, str(output_path / 'comparison_result.html'))
    print("Generated: comparison_result.html")
    
    # Generate Index/Dashboard HTML
    index_html = render_index(
        gdca_metrics=gdca_metrics,
        dca_metrics=dca_metrics,
        backtest_start=backtest_start,
        backtest_end=backtest_end,
    )
    save_html(index_html, str(output_path / 'index.html'))
    print("Generated: index.html")

