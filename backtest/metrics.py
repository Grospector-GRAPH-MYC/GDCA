"""
Performance Metrics Calculation
===============================

This module provides functions for calculating backtest performance metrics.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np


def calculate_drawdown(equity_series: pd.Series) -> pd.DataFrame:
    """
    Calculate drawdown metrics from an equity series.
    
    Args:
        equity_series: Series of portfolio equity values.
        
    Returns:
        DataFrame with Max_Equity and Drawdown_Pct columns.
    """
    df = pd.DataFrame(index=equity_series.index)
    df['Equity'] = equity_series
    df['Max_Equity'] = equity_series.cummax()
    
    # Avoid division by zero
    df['Drawdown_Pct'] = 0.0
    mask = df['Max_Equity'] > 0
    df.loc[mask, 'Drawdown_Pct'] = (
        (df.loc[mask, 'Equity'] - df.loc[mask, 'Max_Equity']) 
        / df.loc[mask, 'Max_Equity'] * 100
    )
    
    return df


def calculate_max_drawdown(equity_data: List[dict]) -> float:
    """
    Calculate maximum drawdown from a list of equity data points.
    
    Args:
        equity_data: List of dicts with 'value' key.
        
    Returns:
        Maximum drawdown percentage (negative value).
    """
    if not equity_data:
        return 0.0
    
    max_equity = 0.0
    max_drawdown = 0.0
    
    for point in equity_data:
        val = point.get('value', 0)
        if val > max_equity:
            max_equity = val
        
        if max_equity > 0:
            drawdown = (val - max_equity) / max_equity * 100
            if drawdown < max_drawdown:
                max_drawdown = drawdown
    
    return max_drawdown


def calculate_metrics(
    total_invested: float,
    final_equity: float,
    final_position: float,
    max_drawdown: float,
    total_fills: int,
) -> Dict[str, Any]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        total_invested: Total amount invested.
        final_equity: Final portfolio equity value.
        final_position: Final BTC position size.
        max_drawdown: Maximum drawdown percentage.
        total_fills: Total number of fills.
        
    Returns:
        Dictionary containing performance metrics.
    """
    net_profit = final_equity - total_invested
    roi = (net_profit / total_invested * 100) if total_invested > 0 else 0.0
    total_trades = total_fills // 2  # Buy + Sell = 1 trade
    
    return {
        'total_invested': total_invested,
        'total_equity': final_equity,
        'net_profit': net_profit,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'total_btc': final_position,
        'total_trades': total_trades,
    }


def calculate_std_dca_metrics(
    std_dca_invested: float,
    std_dca_btc: float,
    current_price: float,
    std_dca_equity_data: List[dict],
) -> Dict[str, Any]:
    """
    Calculate Standard DCA strategy metrics.
    
    Args:
        std_dca_invested: Total amount invested in Standard DCA.
        std_dca_btc: Total BTC accumulated via Standard DCA.
        current_price: Current BTC price.
        std_dca_equity_data: Equity curve data for drawdown calculation.
        
    Returns:
        Dictionary containing Standard DCA metrics.
    """
    std_dca_equity = std_dca_btc * current_price
    std_dca_profit = std_dca_equity - std_dca_invested
    std_dca_roi = (std_dca_profit / std_dca_invested * 100) if std_dca_invested > 0 else 0.0
    std_dca_max_dd = calculate_max_drawdown(std_dca_equity_data)
    
    return {
        'std_dca_invested': std_dca_invested,
        'std_dca_equity': std_dca_equity,
        'std_dca_profit': std_dca_profit,
        'std_dca_roi': std_dca_roi,
        'std_dca_max_drawdown': std_dca_max_dd,
        'std_dca_btc': std_dca_btc,
    }


def format_metrics_for_display(metrics: Dict[str, Any]) -> Dict[str, str]:
    """
    Format metrics dictionary for human-readable display.
    
    Args:
        metrics: Raw metrics dictionary.
        
    Returns:
        Dictionary with formatted string values.
    """
    formatted = {}
    
    for key, value in metrics.items():
        if 'invested' in key or 'equity' in key or 'profit' in key:
            formatted[key] = f"${value:,.2f}"
        elif 'roi' in key or 'drawdown' in key:
            formatted[key] = f"{value:+.2f}%"
        elif 'btc' in key:
            formatted[key] = f"{value:.8f}"
        else:
            formatted[key] = str(value)
    
    return formatted
