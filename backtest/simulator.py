"""
DCA Simulation
==============

This module provides simulation logic for Standard DCA comparison.
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd


@dataclass
class SimulationResult:
    """Results from a DCA simulation."""
    cash_series: List[float]
    invested_series: List[float]
    portfolio_series: List[float]
    holdings_series: List[float]
    final_cash: float
    final_position: float
    total_invested: float


@dataclass
class StandardDCAResult:
    """Results from Standard DCA simulation."""
    btc_accumulated: float
    total_invested: float
    equity_data: List[dict]
    final_equity: float


def simulate_gdca_equity(
    df_bars: pd.DataFrame,
    df_fills: pd.DataFrame,
    dca_amount: float,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    qty_col: str = 'quantity',
    price_col: str = 'price',
    side_col: str = 'side',
) -> SimulationResult:
    """
    Simulate GDCA equity curve with daily deposits.
    
    This simulates a "savings plan" where we deposit DCA_AMOUNT daily
    and track cash, invested total, and portfolio value.
    
    Args:
        df_bars: DataFrame with OHLC price data.
        df_fills: DataFrame with fill/trade data.
        dca_amount: Daily deposit amount.
        start_date: Optional start date for simulation.
        end_date: Optional end date for simulation.
        qty_col: Column name for quantity in fills.
        price_col: Column name for price in fills.
        side_col: Column name for side (buy/sell) in fills.
        
    Returns:
        SimulationResult with equity curve data.
    """
    simul_cash = 0.0
    simul_invested_total = 0.0
    current_pos = 0.0
    last_processed_date = None
    last_processed_fill_idx = 0
    
    cash_series = []
    invested_series = []
    portfolio_series = []
    holdings_series = []
    
    # Sort fills by timestamp
    if not df_fills.empty:
        df_fills = df_fills.sort_values('timestamp')
    
    total_fills = len(df_fills)
    
    for current_time, row in df_bars.iterrows():
        # Check if in date range
        in_range = True
        if start_date and current_time < start_date:
            in_range = False
        if end_date and current_time > end_date:
            in_range = False
        
        # Daily deposit logic
        if in_range:
            row_date = current_time.date()
            if last_processed_date is None or row_date > last_processed_date:
                simul_cash += dca_amount
                simul_invested_total += dca_amount
                last_processed_date = row_date
        
        # Process new fills since last update
        while last_processed_fill_idx < total_fills:
            fill = df_fills.iloc[last_processed_fill_idx]
            fill_ts = fill['timestamp']
            
            if fill_ts > current_time:
                break
            
            # Execute fill
            qty = float(fill[qty_col])
            price = float(fill[price_col])
            side = str(fill[side_col])
            
            cost = price * qty
            
            # Extract commission if available
            comm = _extract_commission(fill)
            
            if 'BUY' in side.upper():
                total_cost = cost + comm
                simul_cash -= total_cost
                current_pos += qty
            elif 'SELL' in side.upper():
                total_proceeds = cost - comm
                simul_cash += total_proceeds
                current_pos -= qty
            
            last_processed_fill_idx += 1
        
        # Calculate portfolio value
        port_val = simul_cash + (current_pos * row['Close'])
        
        cash_series.append(simul_cash)
        invested_series.append(simul_invested_total)
        portfolio_series.append(port_val)
        holdings_series.append(current_pos)
    
    return SimulationResult(
        cash_series=cash_series,
        invested_series=invested_series,
        portfolio_series=portfolio_series,
        holdings_series=holdings_series,
        final_cash=simul_cash,
        final_position=current_pos,
        total_invested=simul_invested_total,
    )


def simulate_standard_dca(
    df_bars: pd.DataFrame,
    dca_amount: float,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
) -> StandardDCAResult:
    """
    Simulate Standard DCA strategy (naive daily buy).
    
    Buys a fixed dollar amount every day at the closing price.
    
    Args:
        df_bars: DataFrame with OHLC price data.
        dca_amount: Daily purchase amount.
        start_date: Optional start date for simulation.
        end_date: Optional end date for simulation.
        
    Returns:
        StandardDCAResult with equity curve and metrics.
    """
    std_dca_btc = 0.0
    std_dca_invested = 0.0
    equity_data = []
    
    for current_time, row in df_bars.iterrows():
        # Check if in date range
        in_range = True
        if start_date and current_time < start_date:
            in_range = False
        if end_date and current_time > end_date:
            in_range = False
        
        current_close = row['Close']
        
        # Buy daily if in range and price is valid
        if in_range and current_close > 0:
            purchased_btc = dca_amount / float(current_close)
            std_dca_btc += purchased_btc
            std_dca_invested += dca_amount
        
        # Calculate current equity
        current_equity = std_dca_btc * float(current_close)
        equity_data.append({
            "time": int(current_time.timestamp()),
            "value": current_equity
        })
    
    final_equity = equity_data[-1]['value'] if equity_data else 0.0
    
    return StandardDCAResult(
        btc_accumulated=std_dca_btc,
        total_invested=std_dca_invested,
        equity_data=equity_data,
        final_equity=final_equity,
    )


def _extract_commission(fill: pd.Series) -> float:
    """
    Extract commission from a fill record.
    
    Args:
        fill: Series representing a single fill.
        
    Returns:
        Commission amount as float.
    """
    comm = 0.0
    
    if 'commission' in fill and pd.notna(fill['commission']):
        try:
            comm_str = str(fill['commission'])
            comm = float(comm_str.split(' ')[0])
        except (ValueError, IndexError):
            comm = 0.0
    elif 'commission_amount' in fill and pd.notna(fill['commission_amount']):
        try:
            comm_str = str(fill['commission_amount'])
            comm = float(comm_str.split(' ')[0])
        except (ValueError, IndexError):
            comm = 0.0
    
    return comm
