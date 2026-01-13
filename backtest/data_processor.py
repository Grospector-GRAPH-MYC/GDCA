"""
Data Processing Utilities
=========================

This module handles loading and preparing data for backtesting.
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np

from nautilus_trader.model.data import BarType, BarSpecification, Bar
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.persistence.catalog import ParquetDataCatalog


def load_bars(
    catalog_path: Path,
    instrument_id: InstrumentId,
    aggregation: BarAggregation = BarAggregation.DAY,
    step: int = 1,
    price_type: PriceType = PriceType.LAST,
) -> List[Bar]:
    """
    Load bar data from a Parquet catalog.
    
    Args:
        catalog_path: Path to the Parquet data catalog.
        instrument_id: The instrument identifier.
        aggregation: Bar aggregation type. Defaults to DAY.
        step: Aggregation step. Defaults to 1.
        price_type: Price type. Defaults to LAST.
        
    Returns:
        List of Bar objects.
    """
    catalog = ParquetDataCatalog(catalog_path)
    
    bar_type = BarType(
        instrument_id,
        BarSpecification(step, aggregation, price_type)
    )
    
    bars = catalog.bars([str(bar_type)])
    return bars


def prepare_bar_dataframe(bars: List[Bar]) -> pd.DataFrame:
    """
    Convert a list of Bar objects to a pandas DataFrame.
    
    Args:
        bars: List of Bar objects from Nautilus Trader.
        
    Returns:
        DataFrame with Date index and OHLC columns.
    """
    df = pd.DataFrame([
        {
            "Date": b.ts_init,
            "Open": b.open.as_double(),
            "High": b.high.as_double(),
            "Low": b.low.as_double(),
            "Close": b.close.as_double(),
        }
        for b in bars
    ])
    
    # Convert epoch nanos to datetime
    df["Date"] = pd.to_datetime(df["Date"], unit="ns", utc=True)
    df.set_index("Date", inplace=True)
    
    # Ensure numeric types
    for col in ['Open', 'High', 'Low', 'Close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def prepare_fills_dataframe(df_fills: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and prepare a fills DataFrame for processing.
    
    Args:
        df_fills: Raw fills DataFrame from Nautilus Trader.
        
    Returns:
        Normalized DataFrame with consistent column names.
    """
    if df_fills.empty:
        return df_fills
    
    # Normalize timestamp column
    if 'ts_init' in df_fills.columns:
        df_fills['timestamp'] = df_fills['ts_init']
    elif 'ts_event' in df_fills.columns:
        df_fills['timestamp'] = df_fills['ts_event']
    
    # Convert to datetime (UTC)
    if 'timestamp' in df_fills.columns:
        if pd.api.types.is_numeric_dtype(df_fills['timestamp']):
            df_fills['timestamp'] = pd.to_datetime(
                df_fills['timestamp'], unit='ns', utc=True
            )
        else:
            df_fills['timestamp'] = pd.to_datetime(
                df_fills['timestamp'], utc=True
            )
    
    # Ensure numeric for price/quantity columns
    for col in ['price', 'last_px', 'quantity', 'qty', 'last_qty']:
        if col in df_fills.columns:
            df_fills[col] = pd.to_numeric(df_fills[col], errors='coerce')
    
    return df_fills.sort_values('timestamp')


def get_side_column(df_fills: pd.DataFrame) -> str:
    """
    Determine which column contains the order side information.
    
    Args:
        df_fills: Fills DataFrame.
        
    Returns:
        Column name for order side.
    """
    if 'order_side' in df_fills.columns:
        return 'order_side'
    return 'side'


def get_quantity_column(df_fills: pd.DataFrame) -> str:
    """
    Determine which column contains the quantity information.
    
    Args:
        df_fills: Fills DataFrame.
        
    Returns:
        Column name for quantity.
    """
    if 'last_qty' in df_fills.columns:
        return 'last_qty'
    elif 'qty' in df_fills.columns:
        return 'qty'
    return 'quantity'


def get_price_column(df_fills: pd.DataFrame) -> str:
    """
    Determine which column contains the price information.
    
    Args:
        df_fills: Fills DataFrame.
        
    Returns:
        Column name for price.
    """
    if 'last_px' in df_fills.columns:
        return 'last_px'
    return 'price'


def clean_for_json(value) -> float:
    """
    Clean a value for JSON serialization.
    
    Handles NaN, Inf, and other non-serializable values.
    
    Args:
        value: Value to clean.
        
    Returns:
        JSON-safe float value.
    """
    if pd.isna(value) or np.isinf(value):
        return 0.0
    return float(value)


def filter_chart_data(
    data: List[dict],
    start_ts: Optional[pd.Timestamp] = None,
    end_ts: Optional[pd.Timestamp] = None,
) -> List[dict]:
    """
    Filter chart data points by timestamp range.
    
    Args:
        data: List of data points with 'time' key (unix timestamp).
        start_ts: Optional start timestamp.
        end_ts: Optional end timestamp.
        
    Returns:
        Filtered list of data points.
    """
    if not start_ts and not end_ts:
        return data
    
    start_unix = int(start_ts.timestamp()) if start_ts else 0
    end_unix = int(end_ts.timestamp()) if end_ts else float('inf')
    
    return [
        point for point in data
        if start_unix <= point.get('time', 0) <= end_unix
    ]
