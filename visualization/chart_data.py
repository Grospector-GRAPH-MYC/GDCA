"""
Chart Data Preparation
======================

This module prepares data for Lightweight Charts visualization.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def clean_value(value) -> float:
    """
    Clean a value for JSON serialization.
    
    Args:
        value: Value to clean (may be NaN, Inf, etc.)
        
    Returns:
        JSON-safe float value.
    """
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def prepare_ohlc_data(
    df_bars: pd.DataFrame,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[dict]:
    """
    Prepare OHLC data for candlestick charts.
    
    Args:
        df_bars: DataFrame with OHLC price data.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        List of OHLC data points for Lightweight Charts.
    """
    data = []
    
    for idx, row in df_bars.iterrows():
        ts = int(idx.timestamp())
        
        # Filter by date range
        if start_ts and idx < start_ts:
            continue
        if end_ts and idx > end_ts:
            continue
        
        data.append({
            "time": ts,
            "open": clean_value(row['Open']),
            "high": clean_value(row['High']),
            "low": clean_value(row['Low']),
            "close": clean_value(row['Close']),
        })
    
    return data


def prepare_equity_data(
    df_equity: pd.DataFrame,
    column: str = 'Equity',
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[dict]:
    """
    Prepare equity/line series data for charts.
    
    Args:
        df_equity: DataFrame with equity data.
        column: Column name to extract. Defaults to 'Equity'.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        List of data points for line/area series.
    """
    data = []
    
    for idx, row in df_equity.iterrows():
        ts = int(idx.timestamp())
        
        if start_ts and idx < start_ts:
            continue
        if end_ts and idx > end_ts:
            continue
        
        if column in row:
            data.append({
                "time": ts,
                "value": clean_value(row[column]),
            })
    
    return data


def prepare_marker_data(
    df_fills: pd.DataFrame,
    side_col: str = 'side',
    price_col: str = 'price',
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[dict]:
    """
    Prepare trade marker data for charts.
    
    Args:
        df_fills: DataFrame with fill/trade data.
        side_col: Column name for order side.
        price_col: Column name for price.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        List of marker data points.
    """
    markers = []
    
    if df_fills.empty:
        return markers
    
    for _, fill in df_fills.iterrows():
        ts = fill['timestamp']
        
        if start_ts and ts < start_ts:
            continue
        if end_ts and ts > end_ts:
            continue
        
        side = str(fill[side_col]).upper()
        price = clean_value(fill[price_col])
        
        if 'BUY' in side:
            markers.append({
                "time": int(ts.timestamp()),
                "position": "belowBar",
                "color": "#10b981",
                "shape": "arrowUp",
                "text": "BUY",
            })
        elif 'SELL' in side:
            markers.append({
                "time": int(ts.timestamp()),
                "position": "aboveBar",
                "color": "#ef4444",
                "shape": "arrowDown",
                "text": "SELL",
            })
    
    return markers


def prepare_indicator_data(
    df_bars: pd.DataFrame,
    column: str,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[dict]:
    """
    Prepare indicator line data for charts.
    
    Args:
        df_bars: DataFrame with indicator data.
        column: Column name of the indicator.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        List of data points for line series.
    """
    data = []
    
    if column not in df_bars.columns:
        return data
    
    for idx, row in df_bars.iterrows():
        ts = int(idx.timestamp())
        
        if start_ts and idx < start_ts:
            continue
        if end_ts and idx > end_ts:
            continue
        
        val = row[column]
        if pd.notna(val) and not np.isinf(val):
            data.append({
                "time": ts,
                "value": float(val),
            })
    
    return data


def prepare_ribbon_data(
    df_bars: pd.DataFrame,
    columns: List[str],
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[List[dict]]:
    """
    Prepare ribbon data (multiple line series) for charts.
    
    Args:
        df_bars: DataFrame with indicator data.
        columns: List of column names for ribbon lines.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        List of line series data.
    """
    return [
        prepare_indicator_data(df_bars, col, start_ts, end_ts)
        for col in columns
        if col in df_bars.columns
    ]


def prepare_zone_data(
    df_bars: pd.DataFrame,
    zone_columns: Dict[str, str],
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> Dict[str, List[dict]]:
    """
    Prepare zone boundary data for charts.
    
    Args:
        df_bars: DataFrame with zone boundary data.
        zone_columns: Dict mapping zone names to column names.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        Dict of zone data series.
    """
    return {
        zone_name: prepare_indicator_data(df_bars, col, start_ts, end_ts)
        for zone_name, col in zone_columns.items()
    }


def prepare_dca_markers(
    df_bars: pd.DataFrame,
    dca_amount: float,
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[dict]:
    """
    Prepare DCA purchase markers for Standard DCA visualization.
    
    Args:
        df_bars: DataFrame with OHLC price data.
        dca_amount: Daily DCA amount.
        start_ts: Optional start timestamp for filtering.
        end_ts: Optional end timestamp for filtering.
        
    Returns:
        List of marker data points for DCA purchases.
    """
    markers = []
    
    for idx, row in df_bars.iterrows():
        if start_ts and idx < start_ts:
            continue
        if end_ts and idx > end_ts:
            continue
        
        markers.append({
            "time": int(idx.timestamp()),
            "position": "belowBar",
            "color": "#8b5cf6",
            "shape": "circle",
            "text": "DCA",
        })
    
    return markers
