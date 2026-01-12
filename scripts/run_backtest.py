from pathlib import Path
from decimal import Decimal
import pandas as pd
import numpy as np

import json
import sys
import os

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import settings

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import StrategyConfig, LoggingConfig
from nautilus_trader.model.data import BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol, ClientId
from nautilus_trader.model.objects import Money, Currency
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.test_kit.providers import TestInstrumentProvider
from nautilus_trader.model.instruments import Instrument

# Import Strategy
from strategies.gdca_strategy import GDCAStrategy, GDCAConfig

def run_backtest():
    # 1. Configuration (Singleton)
    
    # General / Instrument Config
    venue_str = settings.VENUE
    symbol_str_raw = settings.SYMBOL
    
    # Normalize symbol for Nautilus ID (it seems to drop slash in catalog/error message)
    # But keep raw symbol for display if needed. 
    # Use normalized symbol for ID creation to match data.
    symbol_str_normalized = symbol_str_raw.replace("/", "")
    
    venue = Venue(venue_str)
    instrument_id = InstrumentId(Symbol(symbol_str_normalized), venue)
    
    # 2. Engine Setup
    engine_config = BacktestEngineConfig(
        logging=LoggingConfig(log_level="INFO"),
    )
    
    engine = BacktestEngine(config=engine_config)
    
    # Add Strategy
    strategy_config = GDCAConfig(
        instrument_id=str(instrument_id),
        dca_amount=settings.DCA_AMOUNT,
        
        start_short_zone=settings.START_SHORT_ZONE,
        end_short_zone=settings.END_SHORT_ZONE,
        
        start_sell_zone=settings.START_SELL_ZONE,
        end_sell_zone=settings.END_SELL_ZONE,
        
        start_normal_zone=settings.START_NORMAL_ZONE,
        end_normal_zone=settings.END_NORMAL_ZONE,
        
        start_buy_zone=settings.START_BUY_ZONE,
        end_buy_zone=settings.END_BUY_ZONE,
        
        start_long_zone=settings.START_LONG_ZONE,
        end_long_zone=settings.END_LONG_ZONE,
        
        lot_size=settings.LOT_SIZE,
        step_drop_pct=settings.STEP_DROP_PCT,
        take_profit_pct=settings.TAKE_PROFIT_PCT,
        max_positions=settings.MAX_POSITIONS,
        timeframe=settings.TIMEFRAME,
        
        # Backtest Filter
        backtest_start=settings.BACKTEST_START_DATE,
        backtest_end=settings.BACKTEST_END_DATE
    )
    strategy = GDCAStrategy(config=strategy_config)
    engine.add_strategy(strategy)
    
    # 3. Add Venue & Instrument
    from nautilus_trader.model.instruments import CurrencyPair
    from nautilus_trader.model.enums import InstrumentClass, AssetClass
    
    currency_base = Currency.from_str("BTC")
    currency_quote = Currency.from_str("USD")
    
    from nautilus_trader.model.objects import Price, Quantity

    instrument = CurrencyPair(
        instrument_id=instrument_id,
        raw_symbol=Symbol(symbol_str_raw),
        base_currency=currency_base,
        quote_currency=currency_quote,
        price_precision=2,
        size_precision=8, # Bitstamp size precision
        price_increment=Price.from_str("0.01"),
        size_increment=Quantity.from_str("0.00000001"),
        lot_size=None,
        maker_fee=Decimal("0.001"),
        taker_fee=Decimal("0.001"),
        ts_event=0,
        ts_init=0,
    )
    
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING, # Simple netting for backtest
        account_type=AccountType.MARGIN,
        base_currency=currency_quote,
        starting_balances=[Money(1_000_000_000, currency_quote)] # Start with 1B USD (effectively infinite for DCA)
    )
    
    engine.add_instrument(instrument)
    
    # 4. Data
    # Load from catalog
    catalog_path = Path("catalog")
    catalog = ParquetDataCatalog(catalog_path)
    
    bar_type = BarType(
        instrument_id,
        BarSpecification(1, BarAggregation.DAY, PriceType.LAST)
    )
    
    print("Loading data...")
    bars = catalog.bars([str(bar_type)])
    print(f"Loaded {len(bars)} bars.")
    
    # Data is fed entirely to engine to allow indicator warmup.
    # Strategy handles execution filtering internally.

    engine.add_data(bars)
    
    # 5. Run
    print("Running Backtest...")
    engine.run()
    
    # 6. Results
    print("Backtest Complete.")
    engine.trader.generate_account_report(venue)

    # 7. Visualization
    # 7. Visualization
    print("Generating Visualization...")
    # (Plotly imports removed)

    # Extract Data
    # 1. Price Data
    # Convert bars to DataFrame
    df_bars = pd.DataFrame([
        {
            "Date": b.ts_init, # or ts_event, usually ts_init for bar open
            "Open": b.open.as_double(),
            "High": b.high.as_double(),
            "Low": b.low.as_double(),
            "Close": b.close.as_double(),
        }
        for b in bars
    ])
    # Convert epoch nanos to datetime
    df_bars["Date"] = pd.to_datetime(df_bars["Date"], unit="ns", utc=True)
    df_bars.set_index("Date", inplace=True)
    
    # Ensure numeric
    cols = ['Open', 'High', 'Low', 'Close']
    for col in cols:
        df_bars[col] = pd.to_numeric(df_bars[col], errors='coerce')

    # 2. Account Data
    
    df_fills = engine.trader.generate_fills_report()
    
    # Ensure fills timestamp is correct
    if not df_fills.empty:
        # Normalize timestamp column
        if 'ts_init' in df_fills.columns:
            df_fills['timestamp'] = df_fills['ts_init']
        elif 'ts_event' in df_fills.columns:
             df_fills['timestamp'] = df_fills['ts_event']

        # Convert to datetime (UTC)
        if 'timestamp' in df_fills.columns:
             if pd.api.types.is_numeric_dtype(df_fills['timestamp']):
                 df_fills['timestamp'] = pd.to_datetime(df_fills['timestamp'], unit='ns', utc=True)
             else:
                 df_fills['timestamp'] = pd.to_datetime(df_fills['timestamp'], utc=True)
    
    side_col = 'side'
    if 'order_side' in df_fills.columns:
        side_col = 'order_side'
        
    # Ensure numeric for fills
    if not df_fills.empty:
        for col in ['price', 'last_px', 'quantity', 'qty']:
            if col in df_fills.columns:
                df_fills[col] = pd.to_numeric(df_fills[col], errors='coerce')

    # --- DCA Summary Report ---
    total_invested = 0.0
    total_btc = 0.0
    current_value = 0.0
    profit_usd = 0.0
    profit_pct = 0.0
    
    if not df_fills.empty:
        # Determine correct columns
        p_col = 'last_px' if 'last_px' in df_fills.columns else 'price'
        q_col = 'last_qty' if 'last_qty' in df_fills.columns else 'quantity'
        # Fallback for quantity
        if q_col not in df_fills.columns and 'qty' in df_fills.columns: 
             q_col = 'qty'

        # Filter Buys
        # Note: side might be enum or string. Convert to string to be safe.
        buys = df_fills[df_fills[side_col].astype(str).str.upper() == 'BUY']
        
        total_btc = buys[q_col].astype(float).sum() if not buys.empty else 0.0
        total_invested = (buys[q_col].astype(float) * buys[p_col].astype(float)).sum() if not buys.empty else 0.0
        
        last_price = df_bars['Close'].iloc[-1]
        current_value = total_btc * last_price
        
        profit_usd = current_value - total_invested
        profit_pct = (profit_usd / total_invested * 100) if total_invested > 0 else 0.0
        
        if total_invested > 0:
             # Just initial logging, will be overwritten by accurate simulation later
             print(f"Initial Scan: Found {len(buys)} buys vs {len(df_fills)-len(buys)} sells.")
        print("="*60 + "\n")


    # Reuse strategy config for visualization
    viz_config = strategy_config

    # Use Strategy Logic to Calculate Indicators
    print("Calculating indicators using GDCAStrategy logic...")
    df_bars = GDCAStrategy.calculate_visualization_data(df_bars, config=viz_config)

    # Extract Series from DF for easier usage below
    ma_buy = df_bars['MA_Buy']
    ma_strong_buy = df_bars['MA_Strong_Buy']
    ma_sell = df_bars['MA_Sell']
    ma_strong_sell = df_bars['MA_Strong_Sell']
    ma_short = df_bars['MA_Short']
    ma_long = df_bars['MA_Long']

    # Channel Boundaries
    higher_sell = np.maximum(ma_sell, ma_strong_sell)
    lower_sell = np.minimum(ma_sell, ma_strong_sell)
    
    higher_buy = np.maximum(ma_buy, ma_strong_buy)
    lower_buy = np.minimum(ma_buy, ma_strong_buy)
    
    # Channel Gaps (for 10 lines)
    gap_short = (ma_short - higher_sell) / 10
    gap_sell = (higher_sell - lower_sell) / 10
    gap_normal = (lower_sell - higher_buy) / 10
    gap_buy = (higher_buy - ma_strong_buy) / 10
    gap_long = (ma_strong_buy - ma_long) / 10

    # --- Plotly Code Removed (Migrating to Lightweight Charts) ---
    print("Preparing data for Lightweight Charts...")

    
    # --- Equity Curve Reconstruction ---
    # Nautilus reports are great but sometimes we want explicit daily equity for plotting vs Price
    # --- Simulation Setup: Daily Deposit Visualization ---
    # Instead of "Infinite Funding" flat line, we simulate a "Savings Plan".
    # We deposit 'DCA_AMOUNT' Daily.
    # We track "Accumulated Cash" (from deposits) minus "Spent on Buys".
    
    simul_cash = 0.0
    simul_invested_total = 0.0
    dca_sim_amount = float(strategy_config.dca_amount)
    last_processed_date = None
    # --- Standard DCA Simulation Variables ---
    std_dca_btc = 0.0
    std_dca_invested = 0.0
    std_dca_equity_data = [] # List of {'time': '...', 'value': ...}
    
    simul_cash_series = []
    simul_invested_series = []
    portfolio_value_series = []
    simul_holdings_series = [] # New: Track Holdings
    
    # Process Fills to update Cash and Position
    current_pos = 0.0
    
    # Sort fills by time just in case
    if not df_fills.empty:
        df_fills = df_fills.sort_values('timestamp')
        
    last_processed_fill_idx = 0
    total_fills = len(df_fills)
    
    # Determine columns
    qty_col = 'quantity'
    if 'last_qty' in df_fills.columns:
        qty_col = 'last_qty'
    elif 'qty' in df_fills.columns:
        qty_col = 'qty'

        price_col = 'price'
    if 'last_px' in df_fills.columns:
        price_col = 'last_px'
        
    print(f"Using Quantity Col: {qty_col}, Price Col: {price_col}")
    
    # Parse Simulation Date Filter
    start_ts_sim = pd.Timestamp(settings.BACKTEST_START_DATE, tz='UTC') if settings.BACKTEST_START_DATE else None
    end_ts_sim = pd.Timestamp(settings.BACKTEST_END_DATE, tz='UTC') if settings.BACKTEST_END_DATE else None

    for current_time, row in df_bars.iterrows():
        # Check Date Range for Injection/Buying
        in_range = True
        if start_ts_sim and current_time < start_ts_sim: in_range = False
        if end_ts_sim and current_time > end_ts_sim: in_range = False
        
        # Daily Deposit Logic (DCA Plan) - Only if in range
        if in_range:
            row_date = current_time.date()
            if last_processed_date is None or row_date > last_processed_date:
                simul_cash += dca_sim_amount
                simul_invested_total += dca_sim_amount
                last_processed_date = row_date

        # Process new fills since last update
        while last_processed_fill_idx < total_fills:
            fill = df_fills.iloc[last_processed_fill_idx]
            fill_ts = fill['timestamp']
            
            if fill_ts > current_time:
                break
                
            # Execute Fill
            qty = fill[qty_col]
            price = fill[price_col]
            side = str(fill[side_col])
            
            # Cost = Price * Qty
            cost = float(price) * float(qty)
            comm = 0.0 
            if 'commission' in fill and pd.notna(fill['commission']):
                 try:
                     comm_str = str(fill['commission'])
                     comm = float(comm_str.split(' ')[0])
                 except:
                     comm = 0.0
            elif 'commission_amount' in fill and pd.notna(fill['commission_amount']):
                 try:
                     comm_str = str(fill['commission_amount'])
                     comm = float(comm_str.split(' ')[0])
                 except:
                     comm = 0.0

            if 'BUY' in side.upper():
                total_cost = cost + comm
                simul_cash -= total_cost
                current_pos += float(qty)
                
            elif 'SELL' in side.upper():
                total_proceeds = cost - comm
                simul_cash += total_proceeds
                current_pos -= float(qty)
                
            last_processed_fill_idx += 1
            
        # --- Standard DCA Simulation (Naive Daily Buy) ---
        # Buy simply every day at Close price if in range
        # Use the same daily amount as the strategy's DCA_AMOUNT
        current_close_price = row['Close']
        
        if in_range and current_close_price > 0:
            daily_dca_amt = dca_sim_amount 
            purchased_btc = daily_dca_amt / float(current_close_price)
            std_dca_btc += purchased_btc
            std_dca_invested += daily_dca_amt
        
        std_dca_equity = std_dca_btc * float(current_close_price)
        std_dca_equity_data.append({
            "time": int(current_time.timestamp()),
            "value": std_dca_equity
        })

        # Update Simulated Series
        # Portfolio Value = Cash on Hand + (BTC Held * Current Price)
        # Note: simul_cash is essentially "Realized PnL + Unspent injections"
        port_val = simul_cash + (current_pos * row['Close'])
        
        simul_cash_series.append(simul_cash)
        simul_invested_series.append(simul_invested_total)
        portfolio_value_series.append(port_val)
        simul_holdings_series.append(current_pos)

    # Assign to DataFrame
    df_equity = pd.DataFrame(index=df_bars.index) # Re-initialize df_equity here
    df_equity['Close'] = df_bars['Close']
    df_equity['Cash'] = simul_cash_series
    df_equity['Equity'] = portfolio_value_series  # Used as 'Portfolio Value'
    df_equity['BnH'] = simul_invested_series      # Used as 'Invested Capital'
    df_equity['Holdings'] = simul_holdings_series # Used as 'BTC Held'

    
    # --- Drawdown Calculation ---
    # Max Equity based on Portfolio Value? 
    # Or should we track drawdown relative to Invested?
    # Standard: DD from Peak Equity.
    df_equity['Max_Equity'] = df_equity['Equity'].cummax()
    # Avoid div by zero if equity is 0 initially
    df_equity['Drawdown_Pct'] = 0.0
    mask = df_equity['Max_Equity'] > 0
    df_equity.loc[mask, 'Drawdown_Pct'] = (df_equity.loc[mask, 'Equity'] - df_equity.loc[mask, 'Max_Equity']) / df_equity.loc[mask, 'Max_Equity'] * 100
    
    # --- Headline Metrics Calculation ---
    
    # 1. Net Profit
    final_equity = df_equity['Equity'].iloc[-1] # Portfolio Value
    final_invested = df_equity['BnH'].iloc[-1]  # Total Invested
    net_profit = final_equity - final_invested
    net_profit_pct = (net_profit / final_invested * 100) if final_invested > 0 else 0.0
    
    # 2. Max Drawdown
    max_dd_pct = df_equity['Drawdown_Pct'].min() # Negative value e.g. -80%
    final_pos = current_pos
    
    # --- Update Headline Metrics using CORRECT Simulation Data ---
    # Overwrite the temporary placeholders from above to ensure matching
    total_invested = final_invested
    total_equity = final_equity # Renamed for clarity in comparison table
    profit_usd = net_profit
    roi_pct = net_profit_pct # Renamed for clarity in comparison table
    total_btc = final_pos
    
    # Recalculate TradeCount properly by counting Fills / 2
    total_trades = len(df_fills) // 2
    
    # --- Standard DCA Metrics ---
    std_dca_current_value = std_dca_equity # Final day equity
    std_dca_net_profit = std_dca_current_value - std_dca_invested
    std_dca_roi = (std_dca_net_profit / std_dca_invested * 100) if std_dca_invested > 0 else 0.0
    
    # Calculate Max Drawdown for Standard DCA
    std_dca_max_equity = 0
    std_dca_drawdown_max = 0
    for point in std_dca_equity_data:
        val = point['value']
        if val > std_dca_max_equity:
            std_dca_max_equity = val
        
        if std_dca_max_equity > 0:
            dd = (std_dca_max_equity - val) / std_dca_max_equity
            if dd > std_dca_drawdown_max:
                std_dca_drawdown_max = dd
    std_dca_max_dd_pct = -std_dca_drawdown_max * 100 # Make it negative for consistency with GDCA's max_dd_pct

    # Win Rate (Simplification as account_report is not available)
    win_rate_pct = 0.0


    print("\n" + "="*60)
    print("COMPARISON: GDCA STRATEGY vs STANDARD DCA")
    print("="*60)
    
    # Header
    print(f"{'METRIC':<25} | {'GDCA':<15} | {'STANDARD DCA':<15}")
    print("-" * 60)
    
    # Rows
    print(f"{'Total Invested':<25} | ${total_invested:<14,.2f} | ${std_dca_invested:<14,.2f}")
    print(f"{'BTC Holdings':<25} | {total_btc:<15.8f} | {std_dca_btc:<15.8f}")
    print(f"{'Current Value':<25} | ${total_equity:<14,.2f} | ${std_dca_current_value:<14,.2f}")
    print(f"{'Net Profit':<25} | ${profit_usd:<14,.2f} | ${std_dca_net_profit:<14,.2f}")
    print(f"{'ROI':<25} | {roi_pct:<14.2f}% | {std_dca_roi:<14.2f}%")
    print(f"{'Max Drawdown':<25} | {max_dd_pct:<14.2f}% | {std_dca_max_dd_pct:<14.2f}%")
    
    # Win Rate is only relevant for GDCA as active strategy
    print(f"{'Win Rate (Trades)':<25} | {win_rate_pct:<14.2f}% | {'N/A':<15}")

    print("="*60)
    print(f"\nFinal Portfolio Value (GDCA): ${total_equity:,.2f}")
    print(f"Final Return (GDCA): {roi_pct:.2f}%")
    
    # 3. Trade Statistics (Approximate from Fills)
    # We need to pair fills or assume strict closing. 
    # Nautilus `engine.trader.generate_account_report()` has realized PnL column usually? 
    # Or `generate_positions_report()`? 
    # Let's try to infer from "Realized PnL" if available in Account Report, or just use simple approximations.
    # Actually, simpler: Use Strategy generic stats if available? 
    # `engine.trader.generate_account_report(venue)` returns a DataFrame.
    # Let's inspect `account_report` columns? 
    # For now, let's approximate Win Rate from Fills.
    # Alternatively, just use "Total Return" and "Max DD" as primary, and "Total Trades" as fill_count / 2.
    
    total_trades_est = len(df_fills) // 2
    
    # Win Rate (Hard to calculate perfectly without trade list)
    # Let's skimp on Win Rate for now unless we parse `pf_stats` text?
    # Actually, the log output HAD "Win Rate: 0.8". 
    # Nautilus prints this to log! `PnL Statistics (USD)... Win Rate: 0.8`.
    # We can't easily grab that from Python unless we capture stdout or compute it.
    # Let's stick to reliable numbers we have: Net Profit, Return %, Max DD %, Total Trades (Est).
    
    # --- TradingView Lightweight Charts (Output) ---
    import json
    
    print("Serializing data for Lightweight Charts...")
    
    # 1. Prepare OHLC Data with CDC Colors
    ohlc_data = []
    
    # Re-calculate CDC Logic locally for coloring
    ema_fast = df_bars['EMA12']
    ema_slow = df_bars['EMA26']
    close = df_bars['Close']
    
    for t, row in df_bars.iterrows():
        ts = int(t.timestamp())
        
        # Color Logic
        c = row['Close']
        ef = row['EMA12']
        es = row['EMA26']
        
        # Default Gray
        color = '#808080'
        
        bull = ef > es
        bear = ef < es
        
        if bull and c > ef:
             color = '#4caf50' # Green
        elif bear and c > ef and c > es:
             color = '#2196f3' # Blue
        elif bear and c > ef and c < es:
             color = '#00bcd4' # Aqua
        elif bear and c < ef:
             color = '#f44336' # Red
        elif bull and c < ef and c < es:
             color = '#ff9800' # Orange
        elif bull and c < ef and c > es:
             color = '#ffeb3b' # Yellow
             
        ohlc_item = {
            'time': ts,
            'open': float(row['Open']),
            'high': float(row['High']),
            'low': float(row['Low']),
            'close': float(row['Close']),
            'color': color,
            'borderColor': color,
            'wickColor': color
        }
        ohlc_data.append(ohlc_item)

    # 2. Indicators (Lines)
    ema12_data = []
    ema26_data = []
    
    # GDCA Zone Data
    ma_short_data = []
    ma_strong_sell_data = []
    ma_sell_data = []
    ma_buy_data = []
    ma_strong_buy_data = []
    ma_long_data = []
    
    for t, row in df_bars.iterrows():
        ts = int(t.timestamp())
        if pd.notna(row['EMA12']):
            ema12_data.append({'time': ts, 'value': float(row['EMA12'])})
        if pd.notna(row['EMA26']):
            ema26_data.append({'time': ts, 'value': float(row['EMA26'])})
            
        # Serialize Zones
        if pd.notna(ma_short[t]): ma_short_data.append({'time': ts, 'value': float(ma_short[t])})
        if pd.notna(ma_strong_sell[t]): ma_strong_sell_data.append({'time': ts, 'value': float(ma_strong_sell[t])})
        if pd.notna(ma_sell[t]): ma_sell_data.append({'time': ts, 'value': float(ma_sell[t])})
        if pd.notna(ma_buy[t]): ma_buy_data.append({'time': ts, 'value': float(ma_buy[t])})
        if pd.notna(ma_strong_buy[t]): ma_strong_buy_data.append({'time': ts, 'value': float(ma_strong_buy[t])})
        if pd.notna(ma_long[t]): ma_long_data.append({'time': ts, 'value': float(ma_long[t])})

    # 3. Markers (Fills)
    markers = []
    if not df_fills.empty:
        df_fills = df_fills.sort_values('timestamp')
        for idx, fill in df_fills.iterrows():
            ts = int(fill['timestamp'].timestamp())
            side = str(fill[side_col]).upper()
            price = float(fill[price_col])
            qty = float(fill[qty_col])
            
            # Lookup Context (Equity & Zones)
            # Find closest bar timestamp (<= fill timestamp)
            # We can use asof if sorted, or just lookup
            # df_bars index is datetime.
            
            # Use 'asof' to find the latest available bar data at the time of fill
            # We need the index to be sorted.
            
            trade_val = price * qty
            pct_equity = 0.0
            zone_label = "Unknown"
            dca_pct = 0.0 # Default
            
            try:
                # Find index (datetime) closest to fill time
                # Using 'get_indexer' with method='pad' (backward search)
                idx_loc = df_bars.index.get_indexer([fill['timestamp']], method='pad')[0]
                
                if idx_loc != -1:
                    bar_ts = df_bars.index[idx_loc]
                    
                    # 1. Equity Pct
                    # Get Equity from df_equity (aligned with df_bars)
                    curr_equity = df_equity.at[bar_ts, 'Equity']
                    if curr_equity > 0:
                        pct_equity = (trade_val / curr_equity) * 100
                    
                    # 2. Zone Detection & Channel %
                    # Compare price to ribbons at this bar
                    v_short = ma_short.at[bar_ts]
                    v_ssell = ma_strong_sell.at[bar_ts]
                    v_sell = ma_sell.at[bar_ts]
                    v_buy = ma_buy.at[bar_ts]
                    v_sbuy = ma_strong_buy.at[bar_ts]
                    v_long = ma_long.at[bar_ts]
                    
                    # Define Zones (Top, Bottom, Name)
                    # Ordered High to Low assumptions:
                    # Short > Strong Sell > Sell > Buy > Strong Buy > Long
                    
                    channel_pct = 0.0
                    
                    if pd.isna(v_short) or pd.isna(v_long):
                         zone_label = "N/A"
                         channel_pct = 0.0
                         
                    elif price >= v_short:
                        zone_label = "Abv Short"
                        channel_pct = (price - v_short) / v_short * 100
                        dca_pct = 0.0 # Out of zone

                    elif price >= v_ssell:
                        zone_label = "Short Zone"
                        top = v_short
                        bot = v_ssell
                        ratio = (price - bot) / (top - bot)
                        s_val = strategy_config.start_short_zone
                        e_val = strategy_config.end_short_zone
                        # Upper Zone: Entry is Bottom. Start -> End = Bot -> Top.
                        dca_pct = s_val + ratio * (e_val - s_val)
                        
                    elif price >= v_sell:
                        zone_label = "Strong Sell"
                        top = v_ssell
                        bot = v_sell
                        ratio = (price - bot) / (top - bot)
                        s_val = strategy_config.start_sell_zone
                        e_val = strategy_config.end_sell_zone
                        # Upper Zone: Entry is Bottom. Start -> End = Bot -> Top.
                        dca_pct = s_val + ratio * (e_val - s_val)
                        
                    elif price >= v_buy:
                        zone_label = "Normal Buy" 
                        top = v_sell
                        bot = v_buy
                        ratio = (price - bot) / (top - bot)
                        s_val = strategy_config.start_normal_zone
                        e_val = strategy_config.end_normal_zone
                        dca_pct = e_val + ratio * (s_val - e_val)
                        
                    elif price >= v_sbuy:
                        zone_label = "Buy Zone"
                        top = v_buy
                        bot = v_sbuy
                        ratio = (price - bot) / (top - bot)
                        s_val = strategy_config.start_buy_zone
                        e_val = strategy_config.end_buy_zone
                        dca_pct = e_val + ratio * (s_val - e_val)
                        
                    elif price >= v_long:
                        zone_label = "Long Zone"
                        top = v_sbuy
                        bot = v_long
                        ratio = (price - bot) / (top - bot)
                        s_val = strategy_config.start_long_zone
                        e_val = strategy_config.end_long_zone
                        dca_pct = e_val + ratio * (s_val - e_val)
                        
                    else:
                        zone_label = "Bel Long"
                        channel_pct = (price - v_long) / v_long * 100
                        dca_pct = strategy_config.end_long_zone # Cap at max?
                        
            except Exception as e:
                print(f"Error calculating stats for fill at {fill['timestamp']}: {e}")

            is_buy = 'BUY' in side
            color = '#2962FF' if is_buy else '#F23645'
            shape = 'arrowUp' if is_buy else 'arrowDown'
            position = 'belowBar' if is_buy else 'aboveBar'
            
            # Format: 
            # Line 1: SIDE $VALUE 
            # Line 2: QTY BTC (SATS)
            # Line 3: CHANNEL% of ZONE
            # Line 3: CHANNEL% (Based on Config) of ZONE
            qty_sats = int(qty * 100_000_000)
            text = f"{side} ${trade_val:,.2f}<br>{qty:.8f} BTC ({qty_sats:,} Sats)<br>{zone_label}: {dca_pct:.1f}%"
            
            markers.append({
                'time': ts,
                'position': position,
                'color': color,
                'shape': shape,
                'text': '',     # Hide from chart
                'tooltip': text # Store for placeholder
            })

    # 4. Equity Data
    equity_data = []
    bnh_data = []
    cash_data = [] # New: Cash data
    # Using df_equity calculated above
    for t, row in df_equity.iterrows():
        ts = int(t.timestamp())
        # Check for NaN and cast to float
        eq_val = row['Equity']
        if pd.notna(eq_val):
            equity_data.append({'time': ts, 'value': float(eq_val)})
            
        bnh_val = row['BnH']
        if pd.notna(bnh_val):
            bnh_data.append({'time': ts, 'value': float(bnh_val)})

        # New: Cash data
        cash_val = row['Cash']
        if pd.notna(cash_val):
            cash_data.append({'time': ts, 'value': float(cash_val)})

        # New: Holdings data
        holdings_val = row['Holdings']
        if pd.notna(holdings_val):
            # No clean_data needed here as it comes from same index as others
            # But we will clean it downstream
            pass # We'll build the list in a separate loop or just append here?
            # Actually let's just create a new list for it.
            
    holdings_data = []
    for t, row in df_equity.iterrows():
        ts = int(t.timestamp())
        h_val = row['Holdings']
        if pd.notna(h_val):
            holdings_data.append({'time': ts, 'value': float(h_val)})

    # Explicitly Deduplicate and sort again to be absolutely sure
    # (Lightweight charts crashes on unordered or duplicate time)
    def clean_data(data_list):
        # Sort by time
        data_list.sort(key=lambda x: x['time'])
        # Deduplicate (keep last)
        unique_map = {x['time']: x for x in data_list}
        return sorted(unique_map.values(), key=lambda x: x['time'])

    ohlc_data = clean_data(ohlc_data)
    ema12_data = clean_data(ema12_data)
    ema26_data = clean_data(ema26_data)
    markers = clean_data(markers)
    equity_data = clean_data(equity_data)
    bnh_data = clean_data(bnh_data)
    cash_data = clean_data(cash_data) # New: Clean cash data
    holdings_data = clean_data(holdings_data) # New: Clean holdings data
    std_dca_equity_data = clean_data(std_dca_equity_data) # Clean standard DCA data
    
    ma_short_data = clean_data(ma_short_data)
    ma_strong_sell_data = clean_data(ma_strong_sell_data)
    ma_sell_data = clean_data(ma_sell_data)
    ma_buy_data = clean_data(ma_buy_data)
    ma_strong_buy_data = clean_data(ma_strong_buy_data)
    ma_long_data = clean_data(ma_long_data)

    # Serialize
    # Ensure sorted by time
    # ohlc_data.sort(key=lambda x: x['time'])
    # ema12_data.sort(key=lambda x: x['time'])
    # ema26_data.sort(key=lambda x: x['time'])
    # markers.sort(key=lambda x: x['time'])
    # equity_data.sort(key=lambda x: x['time'])
    # bnh_data.sort(key=lambda x: x['time'])
    
    # Deduplicate (Keep last for OHLC/Equity)
    # def dedup(data):
    #     seen = set()
    #     new_data = []
    #     for item in data:
    #         if item['time'] not in seen:
    #             new_data.append(item)
    #             seen.add(item['time'])
    #     return new_data

    # ohlc_data = dedup(ohlc_data)
    # ema12_data = dedup(ema12_data)
    # ema26_data = dedup(ema26_data)
    # Markers don't need dedup, multiple markers per time is allowed
    # equity_data = dedup(equity_data)
    # bnh_data = dedup(bnh_data)

    # 5. Channel Splitting (Python Side)
    # We need 9 intermediate lines for each of the 5 zones.
    # Total 45 lines.
    
    # Pre-calculate steps
    steps = 10
    
    # Helper to interpolate
    def interpolate_zone(top_data, bottom_data):
        # top_data and bottom_data are lists of dicts {'time': ts, 'value': val}
        # We need to match them by time.
        # Use Maps (Dictionaries in Python)
        top_map = {x['time']: x['value'] for x in top_data}
        bottom_map = {x['time']: x['value'] for x in bottom_data}
        
        # We will create 9 lists, one for each intermediate line
        lines = [[] for _ in range(steps - 1)]
        
        # Iterate over all times present in top_data (Limit to valid intersection)
        all_times = sorted(list(set(top_map.keys()) & set(bottom_map.keys())))
        
        for t in all_times:
            top_val = top_map[t]
            bot_val = bottom_map[t]
            
            for i in range(1, steps):
                ratio = i / steps
                val = top_val - (top_val - bot_val) * ratio
                lines[i-1].append({'time': t, 'value': val})
                
        return lines

    # Future Dates Extension (10 Years)
    last_ts = ohlc_data[-1]['time']
    last_close_val = ohlc_data[-1]['close'] # Get last close to use as dummy value
    # 365 days * 10 years = ~3650 days
    # 86400 seconds per day
    future_data = []
    current_ts = last_ts
    for i in range(3650 + 10): # Add 10 extra days for buffer
        current_ts += 86400
        # Use last_close_val so it has a valid number to plot (but will be invisible)
        future_data.append({'time': current_ts, 'value': last_close_val})
        
    # Past Dates Extension (2000 - Start of Data)
    first_ts = ohlc_data[0]['time']
    # Calculate days back to year 2000 (Timestamp 946684801 for Jan 1 2000)
    ts_2000 = 946684800
    
    past_timestamps = []
    curr = first_ts
    while curr > ts_2000:
        curr -= 86400
        past_timestamps.append(curr)
    past_timestamps.reverse()
    
    # Create invisible past data series for timeline extension
    past_data = [{'time': t, 'value': ohlc_data[0]['open']} for t in past_timestamps]

    # --- Project Zones Backward using Log-Linear Regression ---
    # We need to project MA_Base and MA_of_MA back to 2000.
    # Model: log(y) = m*t + c  =>  y = exp(m*t + c)
    
    # 1. Prepare Data for Regression (Drop NaNs)
    df_reg = df_bars.dropna(subset=['MA_Base', 'MA_of_MA']).copy()
    
    # Function to get slope/intercept
    def get_log_linear_params(series_y):
        # Time as independent variable (seconds)
        X = df_reg.index.astype(int) // 10**9 # Convert to seconds
        # Log of price/MA as dependent
        Y = np.log(series_y)
        
        # Polyfit degree 1
        slope, intercept = np.polyfit(X, Y, 1)
        return slope, intercept

    if not df_reg.empty:
        slope_base, icept_base = get_log_linear_params(df_reg['MA_Base'])
        slope_mom, icept_mom = get_log_linear_params(df_reg['MA_of_MA'])
        
        # 2. Generate Past Values
        ma_base_past = []
        ma_mom_past = []
        
        for ts in past_timestamps:
            val_base = np.exp(slope_base * ts + icept_base)
            val_mom = np.exp(slope_mom * ts + icept_mom)
            
            ma_base_past.append({'time': ts, 'value': val_base})
            ma_mom_past.append({'time': ts, 'value': val_mom})
            
        # 3. Calculate Past Channels
        # Convert to arrays for vectorized calc logic or just loop (loop is easier here with basic types)
        # But we need structure for 'interpolate_zone'. 
        # interpolate_zone takes lists of dicts.
        
        # Let's create lists of dicts for each boundary
        
        # Pre-compute boundary lists
        p_ma_short = []
        p_ma_strong_sell = []
        p_ma_sell = []
        p_ma_buy = []
        p_ma_strong_buy = []
        p_ma_long = []
        
        for i, ts in enumerate(past_timestamps):
            base = ma_base_past[i]['value']
            mom = ma_mom_past[i]['value']
            
            # Logic from Lines 193-200
            # Note: We just blindly apply multipliers.
            # Regression gives trend, multipliers give levels.
            
            # ma_buy = MA_Base * manual_buy_multi
            # ...
            # Actually, standard logic:
            
            mb = base * viz_config.manual_buy_multi
            msb = mom * viz_config.manual_buy_multi * viz_config.manual_strong_buy_multi
            
            mse = base * viz_config.manual_sell_multi
            mss = mom * viz_config.manual_sell_multi * viz_config.manual_strong_sell_multi
            
            msh = mss * viz_config.manual_short_multi
            ml = msb * viz_config.manual_long_multi
            
            p_ma_buy.append({'time': ts, 'value': mb})
            p_ma_strong_buy.append({'time': ts, 'value': msb})
            p_ma_sell.append({'time': ts, 'value': mse})
            p_ma_strong_sell.append({'time': ts, 'value': mss})
            p_ma_short.append({'time': ts, 'value': msh})
            p_ma_long.append({'time': ts, 'value': ml})
            
        # 4. Interpolate Ribbons (Past)
        def interpolate_zone_list(top_list, bottom_list):
             # Simplified version of interpolate_zone knowing inputs are aligned lists of dicts
             # But let's re-use interpolate_zone if compatible?
             # interpolate_zone uses Maps. It handles alignment. So it is compatible.
             return interpolate_zone(top_list, bottom_list)

        ribbons_past_payload = {
            'short': interpolate_zone(p_ma_short, p_ma_strong_sell),
            'strong_sell': interpolate_zone(p_ma_strong_sell, p_ma_sell),
            'sell': interpolate_zone(p_ma_sell, p_ma_buy),
            'buy': interpolate_zone(p_ma_buy, p_ma_strong_buy),
            'long': interpolate_zone(p_ma_strong_buy, p_ma_long)
        }
        json_ribbons_past = json.dumps(ribbons_past_payload)
    else:
        # Fallback if regression fails
        json_ribbons_past = "null"

    # Actually, Lightweight Charts ignores points with no value for plotting, but xScale respects them?
    # No, we need a valid series. Let's create an "Invisible" series for the timeline.
    # We will serialize 'future_data' and plot it as a line with 0 opacity.
    
    # 5. Channel Splitting (Python Side)
    # We need 9 intermediate lines for each of the 5 zones.
    # Total 45 lines.
    
    # Pre-calculate steps
    steps = 10
    
    # --- Filter Output Data Lists by Backtest Date Range ---
    # We maintain full history for calculation, but trim for display/JSON
    _trim_start = pd.Timestamp(settings.BACKTEST_START_DATE).timestamp() if settings.BACKTEST_START_DATE else 0
    _trim_end = pd.Timestamp(settings.BACKTEST_END_DATE).timestamp() if settings.BACKTEST_END_DATE else float('inf')
    
    def _trim(dlist):
        return [i for i in dlist if _trim_start <= i['time'] <= _trim_end]
        
    if settings.BACKTEST_START_DATE or settings.BACKTEST_END_DATE:
        print(f"Trimming output data to range: {settings.BACKTEST_START_DATE} - {settings.BACKTEST_END_DATE}")
        ohlc_data = _trim(ohlc_data)
        ema12_data = _trim(ema12_data)
        ema26_data = _trim(ema26_data)
        markers = _trim(markers)
        equity_data = _trim(equity_data)
        bnh_data = _trim(bnh_data)
        cash_data = _trim(cash_data)
        holdings_data = _trim(holdings_data)
        std_dca_equity_data = _trim(std_dca_equity_data)
        
        # Standard DCA Equity Loop Check (redundant if filtered above but safe)
        
        # MA Ribbons Source Data
        ma_short_data = _trim(ma_short_data)
        ma_strong_sell_data = _trim(ma_strong_sell_data)
        ma_sell_data = _trim(ma_sell_data)
        ma_buy_data = _trim(ma_buy_data)
        ma_strong_buy_data = _trim(ma_strong_buy_data)
        ma_long_data = _trim(ma_long_data)


    # Calculate Ribbons
    # 1. Short Zone
    ribbon_short = interpolate_zone(ma_short_data, ma_strong_sell_data)
    # 2. Strong Sell
    ribbon_strong_sell = interpolate_zone(ma_strong_sell_data, ma_sell_data)
    # 3. Sell (Normal)
    ribbon_sell = interpolate_zone(ma_sell_data, ma_buy_data)
    # 4. Buy
    ribbon_buy = interpolate_zone(ma_buy_data, ma_strong_buy_data)
    # 5. Long
    ribbon_long = interpolate_zone(ma_strong_buy_data, ma_long_data)
    
    # Serialize Ribbons - This is a list of lists [line1, line2, ...]
    # We need to flatten or structure it for JS.
    # Let's create a big object: { 'short': [line1...line9], 'strong_sell': ... }
    
    ribbons_payload = {
        'short': ribbon_short,
        'strong_sell': ribbon_strong_sell,
        'sell': ribbon_sell,
        'buy': ribbon_buy,
        'long': ribbon_long
    }
    json_ribbons = json.dumps(ribbons_payload)
    json_future = json.dumps(future_data)
    json_past = json.dumps(past_data)

    # Serialize
    json_ohlc = json.dumps(ohlc_data)
    json_ema12 = json.dumps(ema12_data)
    json_ema26 = json.dumps(ema26_data)
    json_markers = json.dumps(markers)
    json_equity = json.dumps(equity_data)
    json_bnh = json.dumps(bnh_data)
    json_cash = json.dumps(cash_data) # Export Cash Series
    json_holdings = json.dumps(holdings_data) # Export Holdings Series
    json_std_dca_equity = json.dumps(std_dca_equity_data) # Export Standard DCA Equity Series

    
    json_ma_short = json.dumps(ma_short_data)
    json_ma_strong_sell = json.dumps(ma_strong_sell_data)
    json_ma_sell = json.dumps(ma_sell_data)
    json_ma_buy = json.dumps(ma_buy_data)
    json_ma_strong_buy = json.dumps(ma_strong_buy_data)
    json_ma_long = json.dumps(ma_long_data)

    # Prepare metrics for JS
    metrics_payload = {
        "total_invested": total_invested,
        "total_btc": total_btc,
        "total_equity": total_equity,
        "net_profit": profit_usd,
        "roi": roi_pct,
        "max_drawdown": max_dd_pct,
        "win_rate": win_rate_pct,
        "total_trades": total_trades,
        # Standard DCA metrics
        "std_dca_invested": std_dca_invested,
        "std_dca_equity": std_dca_current_value,
        "std_dca_profit": std_dca_net_profit,
        "std_dca_roi": std_dca_roi,
        "std_dca_max_drawdown": std_dca_max_dd_pct
    }
    json_metrics = json.dumps(metrics_payload)
    
    # ========================================
    # HTML GENERATORS
    # ========================================
    
    def generate_strategy_html(
        strategy_name: str,
        strategy_color: str,
        metrics: dict,
        json_ohlc: str,
        json_equity: str,
        json_bnh: str,
        json_cash: str,
        json_holdings: str,
        json_markers: str,
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
        is_gdca: bool = True
    ) -> str:
        """Generate HTML for a single strategy view."""
        
        # Color scheme based on strategy
        primary_color = strategy_color
        gradient_start = "#0a0a0f"
        gradient_end = "#1a1a2e" if is_gdca else "#1a1a1a"
        
        # Metrics for display
        invested = metrics.get('total_invested', 0)
        value = metrics.get('total_equity', 0) if is_gdca else metrics.get('std_dca_equity', 0)
        profit = metrics.get('net_profit', 0) if is_gdca else metrics.get('std_dca_profit', 0)
        roi = metrics.get('roi', 0) if is_gdca else metrics.get('std_dca_roi', 0)
        max_dd = metrics.get('max_drawdown', 0) if is_gdca else metrics.get('std_dca_max_drawdown', 0)
        btc_held = metrics.get('total_btc', 0) if is_gdca else 0
        trades = metrics.get('total_trades', 0) if is_gdca else 'N/A'
        
        profit_class = 'green' if profit >= 0 else 'red'
        
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>{strategy_name} Backtest Results</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, {gradient_start} 0%, {gradient_end} 100%);
            color: #e1e5eb;
            min-height: 100vh;
            overflow-x: hidden;
        }}
        
        .header {{
            background: linear-gradient(90deg, {primary_color}22 0%, transparent 100%);
            border-bottom: 1px solid {primary_color}44;
            padding: 20px 30px;
            display: flex;
            align-items: center;
            gap: 15px;
        }}
        
        .header h1 {{
            font-size: 24px;
            font-weight: 600;
            background: linear-gradient(90deg, {primary_color}, #ffffff);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .header .subtitle {{
            color: #6b7280;
            font-size: 14px;
        }}
        
        .main-container {{
            display: grid;
            grid-template-columns: 320px 1fr;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 80px);
        }}
        
        .metrics-panel {{
            background: rgba(30, 35, 50, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        
        .metric-card {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 16px;
            transition: all 0.3s ease;
        }}
        
        .metric-card:hover {{
            background: rgba(255, 255, 255, 0.06);
            border-color: {primary_color}44;
            transform: translateY(-2px);
        }}
        
        .metric-label {{
            font-size: 12px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 6px;
        }}
        
        .metric-value {{
            font-size: 22px;
            font-weight: 600;
            color: #f3f4f6;
        }}
        
        .metric-value.green {{ color: #10b981; }}
        .metric-value.red {{ color: #ef4444; }}
        
        .metric-sub {{
            font-size: 13px;
            color: #9ca3af;
            margin-top: 4px;
        }}
        
        .charts-container {{
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        .chart-wrapper {{
            background: rgba(20, 24, 35, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }}
        
        .chart-wrapper.price {{ flex: 2; }}
        .chart-wrapper.equity {{ flex: 1; }}
        
        .chart-title {{
            position: absolute;
            top: 12px;
            left: 16px;
            z-index: 10;
            font-size: 13px;
            font-weight: 500;
            color: #9ca3af;
            background: rgba(20, 24, 35, 0.9);
            padding: 6px 12px;
            border-radius: 6px;
            backdrop-filter: blur(4px);
        }}
        
        .controls {{
            position: absolute;
            top: 12px;
            right: 16px;
            z-index: 10;
            display: flex;
            gap: 8px;
        }}
        
        .btn {{
            background: rgba(255, 255, 255, 0.1);
            color: #e1e5eb;
            border: 1px solid rgba(255, 255, 255, 0.1);
            padding: 6px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }}
        
        .btn:hover {{
            background: {primary_color}33;
            border-color: {primary_color};
        }}
        
        .btn.active {{
            background: {primary_color};
            border-color: {primary_color};
        }}
        
        #chart-price, #chart-equity {{ width: 100%; height: 100%; }}
        
        .tooltip-panel {{
            position: absolute;
            top: 45px;
            left: 16px;
            background: rgba(20, 24, 35, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 10px 14px;
            z-index: 20;
            font-size: 12px;
            min-width: 180px;
            pointer-events: none;
        }}
        
        .tooltip-panel .tp-row {{
            display: flex;
            justify-content: space-between;
            margin-bottom: 4px;
        }}
        
        .tooltip-panel .tp-label {{
            color: #6b7280;
        }}
        
        .tooltip-panel .tp-value {{
            color: #f3f4f6;
            font-weight: 500;
        }}
        
        .tooltip-panel .tp-value.green {{ color: #10b981; }}
        .tooltip-panel .tp-value.red {{ color: #ef4444; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{strategy_name}</h1>
        <span class="subtitle">BTC/USD Backtest Results</span>
    </div>
    
    <div class="main-container">
        <div class="metrics-panel">
            <div class="metric-card">
                <div class="metric-label">Total Invested</div>
                <div class="metric-value">${invested:,.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Current Value</div>
                <div class="metric-value">${value:,.2f}</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Net Profit</div>
                <div class="metric-value {profit_class}">${profit:,.2f}</div>
                <div class="metric-sub {profit_class}">{roi:+.2f}% ROI</div>
            </div>
            
            <div class="metric-card">
                <div class="metric-label">Max Drawdown</div>
                <div class="metric-value red">{max_dd:.2f}%</div>
            </div>
            
            {"<div class='metric-card'><div class='metric-label'>BTC Holdings</div><div class='metric-value'>" + f"{btc_held:.8f}" + " BTC</div></div>" if is_gdca else ""}
            
            {"<div class='metric-card'><div class='metric-label'>Total Trades</div><div class='metric-value'>" + str(trades) + "</div></div>" if is_gdca else ""}
            
            <div class="metric-card" style="margin-top: auto;">
                <div class="metric-label">Selected Order</div>
                <div id="order-info" class="metric-value" style="font-size: 14px; color: {primary_color};">Hover over chart</div>
            </div>
        </div>
        
        <div class="charts-container">
            <div class="chart-wrapper price">
                <div class="chart-title">📈 Price Chart</div>
                <div class="controls">
                    <button id="btn-log" class="btn active">Log Scale</button>
                    <button id="btn-reset" class="btn">Reset Zoom</button>
                </div>
                <div id="tooltip-price" class="tooltip-panel" style="display:none;"></div>
                <div id="chart-price"></div>
            </div>
            
            <div class="chart-wrapper equity">
                <div class="chart-title">💰 Portfolio Value</div>
                <div id="tooltip-equity" class="tooltip-panel" style="display:none;"></div>
                <div id="chart-equity"></div>
            </div>
        </div>
    </div>

    <script>
        const ohlcData = {json_ohlc};
        const ema12Data = {json_ema12};
        const ema26Data = {json_ema26};
        const markerData = {json_markers};
        const equityData = {json_equity};
        const bnhData = {json_bnh};
        const cashData = {json_cash};
        const holdingsData = {json_holdings};
        
        const maShortData = {json_ma_short};
        const maStrongSellData = {json_ma_strong_sell};
        const maSellData = {json_ma_sell};
        const maBuyData = {json_ma_buy};
        const maStrongBuyData = {json_ma_strong_buy};
        const maLongData = {json_ma_long};
        
        const ribbons = {json_ribbons};
        const ribbonsPast = {json_ribbons_past};
        const futureData = {json_future};
        const pastData = {json_past};

        // Price Chart
        const priceContainer = document.getElementById('chart-price');
        const priceChart = LightweightCharts.createChart(priceContainer, {{
            localization: {{ locale: 'en-US' }},
            layout: {{ background: {{ color: 'transparent' }}, textColor: '#9ca3af' }},
            grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
            timeScale: {{ borderColor: '#374151', timeVisible: true }},
            rightPriceScale: {{ borderColor: '#374151', mode: LightweightCharts.PriceScaleMode.Logarithmic }},
            crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }}
        }});

        // Zone Ribbons
        const steps = 10;
        function drawRibbon(ribbonLists, boundaryData, colorHex, style) {{
            const lineStyle = style !== undefined ? style : 0;
            if (ribbonLists) {{
                for (const lineData of ribbonLists) {{
                    const lineSeries = priceChart.addLineSeries({{
                        color: colorHex, lineWidth: 1, 
                        crosshairMarkerVisible: false, priceLineVisible: false, lastValueVisible: false, lineStyle: lineStyle
                    }});
                    lineSeries.setData(lineData);
                }}
            }}
            if (boundaryData) {{
                const lineSeries = priceChart.addLineSeries({{
                    color: colorHex, lineWidth: 2, title: '', lastValueVisible: false, lineStyle: lineStyle, crosshairMarkerVisible: false
                }});
                lineSeries.setData(boundaryData);
            }}
        }}

        drawRibbon(ribbons.short, maShortData, '#801922');
        drawRibbon(ribbons.strong_sell, maStrongSellData, '#b22833');
        drawRibbon(ribbons.sell, maSellData, '#f57f17');
        drawRibbon(ribbons.buy, maBuyData, '#1b5e20');
        drawRibbon(ribbons.long, maStrongBuyData, '#00332a');
        
        const lineLong = priceChart.addLineSeries({{ color: '#00332a', lineWidth: 2, lastValueVisible: false, crosshairMarkerVisible: false }});
        lineLong.setData(maLongData);

        if (ribbonsPast) {{
            drawRibbon(ribbonsPast.short, null, '#801922', 2);
            drawRibbon(ribbonsPast.strong_sell, null, '#b22833', 2);
            drawRibbon(ribbonsPast.sell, null, '#f57f17', 2);
            drawRibbon(ribbonsPast.buy, null, '#1b5e20', 2);
            drawRibbon(ribbonsPast.long, null, '#00332a', 2);
        }}

        if (futureData && futureData.length > 0) {{
            const futureSeries = priceChart.addLineSeries({{ color: 'rgba(0,0,0,0)', lineWidth: 0, priceLineVisible: false, lastValueVisible: false }});
            futureSeries.setData(futureData);
        }}
        
        if (pastData && pastData.length > 0) {{
            const pastSeries = priceChart.addLineSeries({{ color: 'rgba(0,0,0,0)', lineWidth: 0, priceLineVisible: false, lastValueVisible: false }});
            pastSeries.setData(pastData);
        }}

        const line12 = priceChart.addLineSeries({{ color: '#f23645', lineWidth: 1, title: 'EMA 12' }});
        line12.setData(ema12Data);
        
        const line26 = priceChart.addLineSeries({{ color: '#2962FF', lineWidth: 1, title: 'EMA 26' }});
        line26.setData(ema26Data);

        const candleSeries = priceChart.addCandlestickSeries({{
            upColor: '#10b981', downColor: '#ef4444', borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#ef4444'
        }});
        candleSeries.setData(ohlcData);
        candleSeries.setMarkers(markerData);

        // Equity Chart
        const equityChart = LightweightCharts.createChart(document.getElementById('chart-equity'), {{
            localization: {{ locale: 'en-US' }},
            layout: {{ background: {{ color: 'transparent' }}, textColor: '#9ca3af' }},
            grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
            rightPriceScale: {{ borderColor: '#374151' }},
            timeScale: {{ borderColor: '#374151' }}
        }});

        const equitySeries = equityChart.addAreaSeries({{
            topColor: '{primary_color}40', bottomColor: '{primary_color}05', lineColor: '{primary_color}', lineWidth: 2
        }});
        equitySeries.setData(equityData);
        
        {"const cashSeries = equityChart.addAreaSeries({ topColor: '#10b98140', bottomColor: '#10b98105', lineColor: '#10b981', lineWidth: 1 }); cashSeries.setData(cashData);" if is_gdca else ""}
        
        const bnhSeries = equityChart.addLineSeries({{ color: '#f59e0b', lineWidth: 2, lineStyle: 2 }});
        bnhSeries.setData(bnhData);

        {"const holdingsSeries = equityChart.addAreaSeries({ topColor: '#fbbf2440', bottomColor: '#fbbf2405', lineColor: '#fbbf24', lineWidth: 1, priceScaleId: 'left' }); holdingsSeries.setData(holdingsData);" if is_gdca else ""}

        equitySeries.setMarkers(markerData);
        equityChart.timeScale().fitContent();

        // Sync Charts
        function syncVisibleRange(source, target) {{
            const visibleRange = source.timeScale().getVisibleRange();
            if (visibleRange) target.timeScale().setVisibleRange(visibleRange);
        }}
        priceChart.timeScale().subscribeVisibleTimeRangeChange(() => syncVisibleRange(priceChart, equityChart));
        equityChart.timeScale().subscribeVisibleTimeRangeChange(() => syncVisibleRange(equityChart, priceChart));

        // Resize
        new ResizeObserver(entries => {{
            for (let entry of entries) {{
                const {{ width, height }} = entry.contentRect;
                if (entry.target.id === 'chart-price') priceChart.applyOptions({{ width, height }});
                if (entry.target.id === 'chart-equity') equityChart.applyOptions({{ width, height }});
            }}
        }}).observe(priceContainer);
        new ResizeObserver(entries => {{
            for (let entry of entries) {{
                const {{ width, height }} = entry.contentRect;
                equityChart.applyOptions({{ width, height }});
            }}
        }}).observe(document.getElementById('chart-equity'));

        // Tooltips
        const orderInfoEl = document.getElementById('order-info');
        const tooltipPrice = document.getElementById('tooltip-price');
        const tooltipEquity = document.getElementById('tooltip-equity');
        const markerMap = new Map();
        if (markerData) markerData.forEach(m => markerMap.set(m.time, m));
        
        // Create data maps for O(1) lookup
        const ohlcMap = new Map();
        ohlcData.forEach(d => ohlcMap.set(d.time, d));
        const equityMap = new Map();
        equityData.forEach(d => equityMap.set(d.time, d));
        const bnhMap = new Map();
        bnhData.forEach(d => bnhMap.set(d.time, d));
        const cashMap = new Map();
        cashData.forEach(d => cashMap.set(d.time, d));
        const holdingsMap = new Map();
        holdingsData.forEach(d => holdingsMap.set(d.time, d));
        
        // Price Chart Crosshair
        priceChart.subscribeCrosshairMove(param => {{
            if (!param.time) {{
                tooltipPrice.style.display = 'none';
                orderInfoEl.innerHTML = 'Hover over chart';
                return;
            }}
            
            const ohlc = ohlcMap.get(param.time);
            if (ohlc) {{
                const date = new Date(param.time * 1000);
                const dateStr = date.toLocaleDateString('en-US', {{calendar: 'gregory', month: 'short', day: 'numeric', year: 'numeric'}});
                const change = ((ohlc.close - ohlc.open) / ohlc.open * 100);
                const changeClass = change >= 0 ? 'green' : 'red';
                tooltipPrice.innerHTML = `
                    <div class="tp-row"><span class="tp-label">Date</span><span class="tp-value">${{dateStr}}</span></div>
                    <div class="tp-row"><span class="tp-label">Open</span><span class="tp-value">${{ohlc.open.toLocaleString('en-US', {{minimumFractionDigits: 2}})}}</span></div>
                    <div class="tp-row"><span class="tp-label">High</span><span class="tp-value">${{ohlc.high.toLocaleString('en-US', {{minimumFractionDigits: 2}})}}</span></div>
                    <div class="tp-row"><span class="tp-label">Low</span><span class="tp-value">${{ohlc.low.toLocaleString('en-US', {{minimumFractionDigits: 2}})}}</span></div>
                    <div class="tp-row"><span class="tp-label">Close</span><span class="tp-value">${{ohlc.close.toLocaleString('en-US', {{minimumFractionDigits: 2}})}}</span></div>
                    <div class="tp-row"><span class="tp-label">Change</span><span class="tp-value ${{changeClass}}">${{change >= 0 ? '+' : ''}}${{change.toFixed(2)}}%</span></div>
                `;
                tooltipPrice.style.display = 'block';
            }} else {{
                tooltipPrice.style.display = 'none';
            }}
            
            // Order info
            const marker = markerMap.get(param.time);
            if (marker) {{
                orderInfoEl.innerHTML = marker.tooltip;
                orderInfoEl.style.color = marker.color;
            }} else {{
                orderInfoEl.innerHTML = '-';
                orderInfoEl.style.color = '{primary_color}';
            }}
        }});
        
        // Equity Chart Crosshair
        equityChart.subscribeCrosshairMove(param => {{
            if (!param.time) {{
                tooltipEquity.style.display = 'none';
                return;
            }}
            
            const equity = equityMap.get(param.time);
            const bnh = bnhMap.get(param.time);
            const cash = cashMap.get(param.time);
            const holdings = holdingsMap.get(param.time);
            
            if (equity) {{
                const date = new Date(param.time * 1000);
                const dateStr = date.toLocaleDateString('en-US', {{calendar: 'gregory', month: 'short', day: 'numeric', year: 'numeric'}});
                const invested = bnh ? bnh.value : 0;
                const profit = equity.value - invested;
                const profitClass = profit >= 0 ? 'green' : 'red';
                
                let html = `
                    <div class="tp-row"><span class="tp-label">Date</span><span class="tp-value">${{dateStr}}</span></div>
                    <div class="tp-row"><span class="tp-label">Portfolio</span><span class="tp-value">${{equity.value.toLocaleString('en-US', {{style: 'currency', currency: 'USD'}})}}</span></div>
                    <div class="tp-row"><span class="tp-label">Invested</span><span class="tp-value">${{invested.toLocaleString('en-US', {{style: 'currency', currency: 'USD'}})}}</span></div>
                    <div class="tp-row"><span class="tp-label">Profit</span><span class="tp-value ${{profitClass}}">${{profit.toLocaleString('en-US', {{style: 'currency', currency: 'USD'}})}}</span></div>
                `;
                if (cash && cash.value > 0) {{
                    html += `<div class="tp-row"><span class="tp-label">Cash</span><span class="tp-value">${{cash.value.toLocaleString('en-US', {{style: 'currency', currency: 'USD'}})}}</span></div>`;
                }}
                if (holdings && holdings.value > 0) {{
                    html += `<div class="tp-row"><span class="tp-label">BTC</span><span class="tp-value">${{holdings.value.toFixed(4)}} BTC</span></div>`;
                }}
                tooltipEquity.innerHTML = html;
                tooltipEquity.style.display = 'block';
            }} else {{
                tooltipEquity.style.display = 'none';
            }}
        }});

        // Controls
        const btnLog = document.getElementById('btn-log');
        let isLog = true;
        btnLog.addEventListener('click', () => {{
            isLog = !isLog;
            priceChart.priceScale('right').applyOptions({{ mode: isLog ? LightweightCharts.PriceScaleMode.Logarithmic : LightweightCharts.PriceScaleMode.Normal }});
            btnLog.classList.toggle('active', isLog);
            btnLog.innerText = isLog ? 'Log Scale' : 'Linear';
        }});
        
        document.getElementById('btn-reset').addEventListener('click', () => {{
            if (ohlcData.length > 0) {{
                priceChart.timeScale().setVisibleRange({{ from: ohlcData[0].time, to: ohlcData[ohlcData.length - 1].time }});
            }}
        }});
        
        setTimeout(() => {{
            if (ohlcData.length > 0) {{
                priceChart.timeScale().setVisibleRange({{ from: ohlcData[0].time, to: ohlcData[ohlcData.length - 1].time }});
            }}
        }}, 100);
        <div class="footer">
            <div class="brand">
                <span class="project-name">M Y C</span>
                <span class="divider"></span>
                <span class="creator">G R A P H</span>
            </div>
            <div class="system-info">
                Enterprise Edition Ready • Powered by Nautilus Trader
            </div>
        </div>
    </div>
    
    <style>
        .footer {
            margin-top: 60px;
            text-align: center;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 24px;
            margin-bottom: 24px;
        }

        .brand {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 8px;
            font-weight: 700;
            letter-spacing: 2px;
        }

        .project-name {
            background: linear-gradient(90deg, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 13px;
        }

        .creator {
            color: #4b5563;
            font-size: 11px;
        }

        .divider {
            width: 3px;
            height: 3px;
            background: #4b5563;
            border-radius: 50%;
        }

        .system-info {
            color: #4b5563;
            font-size: 10px;
            letter-spacing: 0.5px;
        }
    </style>
</body>
</html>'''

    def generate_comparison_html(
        gdca_metrics: dict,
        dca_metrics: dict,
        json_equity_gdca: str,
        json_equity_dca: str,
        json_bnh: str
    ) -> str:
        """Generate HTML for side-by-side comparison."""
        
        # Determine winner
        gdca_roi = gdca_metrics.get('roi', 0)
        dca_roi = dca_metrics.get('std_dca_roi', 0)
        gdca_wins = gdca_roi > dca_roi
        
        return f'''<!DOCTYPE html>
<html>
<head>
    <title>Strategy Comparison | GDCA vs Standard DCA</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #0a0a0f 0%, #111827 100%);
            color: #e1e5eb;
            min-height: 100vh;
            padding: 24px;
        }}
        
        .header {{
            text-align: center;
            margin-bottom: 30px;
        }}
        
        .header h1 {{
            font-size: 28px;
            font-weight: 700;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .comparison-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 24px;
            margin-bottom: 24px;
        }}
        
        .strategy-card {{
            background: rgba(30, 35, 50, 0.6);
            backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 24px;
            position: relative;
            transition: all 0.3s ease;
        }}
        
        .strategy-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
        }}
        
        .strategy-card.gdca {{ border-top: 3px solid #3b82f6; }}
        .strategy-card.dca {{ border-top: 3px solid #8b5cf6; }}
        
        .strategy-card.winner {{
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.2);
            border-color: rgba(16, 185, 129, 0.3);
        }}
        
        .winner-badge {{
            position: absolute;
            top: -12px;
            right: 20px;
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        
        .strategy-title {{
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        
        .strategy-title.gdca {{ color: #3b82f6; }}
        .strategy-title.dca {{ color: #8b5cf6; }}
        
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 16px;
        }}
        
        .metric {{
            background: rgba(255, 255, 255, 0.03);
            border-radius: 10px;
            padding: 14px;
        }}
        
        .metric-label {{
            font-size: 11px;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        
        .metric-value {{
            font-size: 18px;
            font-weight: 600;
        }}
        
        .metric-value.green {{ color: #10b981; }}
        .metric-value.red {{ color: #ef4444; }}
        
        .chart-section {{
            background: rgba(20, 24, 35, 0.8);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 16px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        
        .chart-section h3 {{
            font-size: 16px;
            font-weight: 500;
            color: #9ca3af;
            margin-bottom: 16px;
        }}
        
        .charts-row {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            height: 300px;
        }}
        
        .chart-box {{
            background: rgba(15, 18, 25, 0.6);
            border-radius: 12px;
            overflow: hidden;
            position: relative;
        }}
        
        .chart-label {{
            position: absolute;
            top: 10px;
            left: 14px;
            font-size: 12px;
            font-weight: 500;
            z-index: 10;
            padding: 4px 10px;
            border-radius: 4px;
            background: rgba(0,0,0,0.5);
        }}
        
        .chart-label.gdca {{ color: #3b82f6; }}
        .chart-label.dca {{ color: #8b5cf6; }}
        
        #chart-gdca, #chart-dca, #chart-overlay {{ width: 100%; height: 100%; }}
        
        .overlay-section {{
            height: 350px;
        }}
        
        #chart-overlay {{ height: 300px; }}
        
        .tooltip-panel {{
            position: absolute;
            top: 35px;
            left: 14px;
            background: rgba(20, 24, 35, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            padding: 10px 14px;
            z-index: 20;
            font-size: 12px;
            min-width: 180px;
            pointer-events: none;
        }}
        
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 Strategy Comparison</h1>
        <p style="color: #6b7280; margin-top: 8px;">GDCA vs Standard DCA • BTC/USD</p>
    </div>
    
    <div class="comparison-grid">
        <div class="strategy-card gdca {'winner' if gdca_wins else ''}">
            {'<div class="winner-badge">🏆 WINNER</div>' if gdca_wins else ''}
            <div class="strategy-title gdca">📈 GDCA Strategy</div>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Total Invested</div>
                    <div class="metric-value">${gdca_metrics['total_invested']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Current Value</div>
                    <div class="metric-value">${gdca_metrics['total_equity']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Net Profit</div>
                    <div class="metric-value {'green' if gdca_metrics['net_profit'] >= 0 else 'red'}">${gdca_metrics['net_profit']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ROI</div>
                    <div class="metric-value {'green' if gdca_metrics['roi'] >= 0 else 'red'}">{gdca_metrics['roi']:+.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value red">{gdca_metrics['max_drawdown']:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">BTC Holdings</div>
                    <div class="metric-value">{gdca_metrics['total_btc']:.4f}</div>
                </div>
            </div>
        </div>
        
        <div class="strategy-card dca {'winner' if not gdca_wins else ''}">
            {'<div class="winner-badge">🏆 WINNER</div>' if not gdca_wins else ''}
            <div class="strategy-title dca">📊 Standard DCA</div>
            <div class="metrics-grid">
                <div class="metric">
                    <div class="metric-label">Total Invested</div>
                    <div class="metric-value">${dca_metrics['std_dca_invested']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Current Value</div>
                    <div class="metric-value">${dca_metrics['std_dca_equity']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Net Profit</div>
                    <div class="metric-value {'green' if dca_metrics['std_dca_profit'] >= 0 else 'red'}">${dca_metrics['std_dca_profit']:,.0f}</div>
                </div>
                <div class="metric">
                    <div class="metric-label">ROI</div>
                    <div class="metric-value {'green' if dca_metrics['std_dca_roi'] >= 0 else 'red'}">{dca_metrics['std_dca_roi']:+.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Max Drawdown</div>
                    <div class="metric-value red">{dca_metrics['std_dca_max_drawdown']:.1f}%</div>
                </div>
                <div class="metric">
                    <div class="metric-label">Strategy</div>
                    <div class="metric-value" style="font-size: 14px;">Fixed Monthly Buy</div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="chart-section">
        <h3>📈 Individual Performance</h3>
        <div class="charts-row">
            <div class="chart-box">
                <div class="chart-label gdca">GDCA Portfolio</div>
                <div id="tooltip-gdca" class="tooltip-panel" style="display:none;"></div>
                <div id="chart-gdca"></div>
            </div>
            <div class="chart-box">
                <div class="chart-label dca">Standard DCA Portfolio</div>
                <div id="tooltip-dca" class="tooltip-panel" style="display:none;"></div>
                <div id="chart-dca"></div>
            </div>
        </div>
    </div>
    
    <div class="chart-section overlay-section">
        <h3>🔄 Combined Overlay</h3>
        <div id="tooltip-overlay" class="tooltip-panel" style="display:none;"></div>
        <div id="chart-overlay"></div>
    </div>

    <script>
        const equityGdca = {json_equity_gdca};
        const equityDca = {json_equity_dca};
        const bnhData = {json_bnh};

        const chartOptions = {{
            localization: {{ locale: 'en-US' }},
            layout: {{ background: {{ color: 'transparent' }}, textColor: '#9ca3af' }},
            grid: {{ vertLines: {{ color: '#1f2937' }}, horzLines: {{ color: '#1f2937' }} }},
            rightPriceScale: {{ borderColor: '#374151' }},
            timeScale: {{ borderColor: '#374151' }}
        }};

        // GDCA Chart
        const gdcaChart = LightweightCharts.createChart(document.getElementById('chart-gdca'), chartOptions);
        const gdcaSeries = gdcaChart.addAreaSeries({{
            topColor: '#3b82f640', bottomColor: '#3b82f605', lineColor: '#3b82f6', lineWidth: 2
        }});
        gdcaSeries.setData(equityGdca);
        const gdcaBnh = gdcaChart.addLineSeries({{ color: '#f59e0b', lineWidth: 1, lineStyle: 2 }});
        gdcaBnh.setData(bnhData);
        gdcaChart.timeScale().fitContent();

        // DCA Chart
        const dcaChart = LightweightCharts.createChart(document.getElementById('chart-dca'), chartOptions);
        const dcaSeries = dcaChart.addAreaSeries({{
            topColor: '#8b5cf640', bottomColor: '#8b5cf605', lineColor: '#8b5cf6', lineWidth: 2
        }});
        dcaSeries.setData(equityDca);
        const dcaBnh = dcaChart.addLineSeries({{ color: '#f59e0b', lineWidth: 1, lineStyle: 2 }});
        dcaBnh.setData(bnhData);
        dcaChart.timeScale().fitContent();

        // Overlay Chart
        const overlayChart = LightweightCharts.createChart(document.getElementById('chart-overlay'), chartOptions);
        const overlayGdca = overlayChart.addLineSeries({{ color: '#3b82f6', lineWidth: 2, title: 'GDCA' }});
        overlayGdca.setData(equityGdca);
        const overlayDca = overlayChart.addLineSeries({{ color: '#8b5cf6', lineWidth: 2, title: 'Standard DCA' }});
        overlayDca.setData(equityDca);
        const overlayBnh = overlayChart.addLineSeries({{ color: '#f59e0b', lineWidth: 1, lineStyle: 2, title: 'Invested' }});
        overlayBnh.setData(bnhData);
        overlayChart.timeScale().fitContent();

        // Sync all charts
        function syncCharts(source, targets) {{
            const range = source.timeScale().getVisibleRange();
            if (range) targets.forEach(t => t.timeScale().setVisibleRange(range));
        }}
        
        gdcaChart.timeScale().subscribeVisibleTimeRangeChange(() => syncCharts(gdcaChart, [dcaChart, overlayChart]));
        dcaChart.timeScale().subscribeVisibleTimeRangeChange(() => syncCharts(dcaChart, [gdcaChart, overlayChart]));
        overlayChart.timeScale().subscribeVisibleTimeRangeChange(() => syncCharts(overlayChart, [gdcaChart, dcaChart]));

        // Resize
        new ResizeObserver(() => {{
            const gdcaEl = document.getElementById('chart-gdca');
            const dcaEl = document.getElementById('chart-dca');
            const overlayEl = document.getElementById('chart-overlay');
            gdcaChart.applyOptions({{ width: gdcaEl.clientWidth, height: gdcaEl.clientHeight }});
            dcaChart.applyOptions({{ width: dcaEl.clientWidth, height: dcaEl.clientHeight }});
            overlayChart.applyOptions({{ width: overlayEl.clientWidth, height: overlayEl.clientHeight }});
        }}).observe(document.body);
        
        // Tooltips
        const tooltipGdca = document.getElementById('tooltip-gdca');
        const tooltipDca = document.getElementById('tooltip-dca');
        const tooltipOverlay = document.getElementById('tooltip-overlay');
        
        // Data maps for O(1) lookup
        const gdcaMap = new Map();
        equityGdca.forEach(d => gdcaMap.set(d.time, d));
        const dcaMap = new Map();
        equityDca.forEach(d => dcaMap.set(d.time, d));
        const bnhMap = new Map();
        bnhData.forEach(d => bnhMap.set(d.time, d));
        
        function formatTooltip(date, value, invested, label) {{
            const profit = value - invested;
            const profitClass = profit >= 0 ? 'green' : 'red';
            const roi = invested > 0 ? ((value - invested) / invested * 100) : 0;
            return `
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#6b7280;">Date</span><span style="color:#f3f4f6;font-weight:500;">${{date}}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#6b7280;">${{label}}</span><span style="color:#f3f4f6;font-weight:500;">${{value.toLocaleString('en-US', {{style:'currency',currency:'USD'}})}}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#6b7280;">Invested</span><span style="color:#f3f4f6;font-weight:500;">${{invested.toLocaleString('en-US', {{style:'currency',currency:'USD'}})}}</span></div>
                <div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#6b7280;">Profit</span><span style="color:${{profit >= 0 ? '#10b981' : '#ef4444'}};font-weight:500;">${{profit.toLocaleString('en-US', {{style:'currency',currency:'USD'}})}}</span></div>
                <div style="display:flex;justify-content:space-between;"><span style="color:#6b7280;">ROI</span><span style="color:${{roi >= 0 ? '#10b981' : '#ef4444'}};font-weight:500;">${{roi >= 0 ? '+' : ''}}${{roi.toFixed(2)}}%</span></div>
            `;
        }}
        
        // GDCA Chart Crosshair
        gdcaChart.subscribeCrosshairMove(param => {{
            if (!param.time) {{ tooltipGdca.style.display = 'none'; return; }}
            const gdca = gdcaMap.get(param.time);
            const bnh = bnhMap.get(param.time);
            if (gdca) {{
                const date = new Date(param.time * 1000).toLocaleDateString('en-US', {{calendar: 'gregory', month:'short',day:'numeric',year:'numeric'}});
                tooltipGdca.innerHTML = formatTooltip(date, gdca.value, bnh ? bnh.value : 0, 'GDCA Value');
                tooltipGdca.style.display = 'block';
            }} else {{ tooltipGdca.style.display = 'none'; }}
        }});
        
        // DCA Chart Crosshair
        dcaChart.subscribeCrosshairMove(param => {{
            if (!param.time) {{ tooltipDca.style.display = 'none'; return; }}
            const dca = dcaMap.get(param.time);
            const bnh = bnhMap.get(param.time);
            if (dca) {{
                const date = new Date(param.time * 1000).toLocaleDateString('en-US', {{calendar: 'gregory', month:'short',day:'numeric',year:'numeric'}});
                tooltipDca.innerHTML = formatTooltip(date, dca.value, bnh ? bnh.value : 0, 'DCA Value');
                tooltipDca.style.display = 'block';
            }} else {{ tooltipDca.style.display = 'none'; }}
        }});
        
        // Overlay Chart Crosshair
        overlayChart.subscribeCrosshairMove(param => {{
            if (!param.time) {{ tooltipOverlay.style.display = 'none'; return; }}
            const gdca = gdcaMap.get(param.time);
            const dca = dcaMap.get(param.time);
            const bnh = bnhMap.get(param.time);
            if (gdca || dca) {{
                const date = new Date(param.time * 1000).toLocaleDateString('en-US', {{calendar: 'gregory', month:'short',day:'numeric',year:'numeric'}});
                const invested = bnh ? bnh.value : 0;
                let html = `<div style="display:flex;justify-content:space-between;margin-bottom:6px;"><span style="color:#6b7280;">Date</span><span style="color:#f3f4f6;font-weight:500;">${{date}}</span></div>`;
                if (gdca) {{
                    const profit = gdca.value - invested;
                    html += `<div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#3b82f6;">● GDCA</span><span style="color:#f3f4f6;font-weight:500;">${{gdca.value.toLocaleString('en-US', {{style:'currency',currency:'USD'}})}}</span></div>`;
                }}
                if (dca) {{
                    html += `<div style="display:flex;justify-content:space-between;margin-bottom:4px;"><span style="color:#8b5cf6;">● DCA</span><span style="color:#f3f4f6;font-weight:500;">${{dca.value.toLocaleString('en-US', {{style:'currency',currency:'USD'}})}}</span></div>`;
                }}
                html += `<div style="display:flex;justify-content:space-between;"><span style="color:#f59e0b;">● Invested</span><span style="color:#f3f4f6;font-weight:500;">${{invested.toLocaleString('en-US', {{style:'currency',currency:'USD'}})}}</span></div>`;
                tooltipOverlay.innerHTML = html;
                tooltipOverlay.style.display = 'block';
            }} else {{ tooltipOverlay.style.display = 'none'; }}
        }});
        
        // Resize
        new ResizeObserver(() => {{
            const gdcaEl = document.getElementById('chart-gdca');
            const dcaEl = document.getElementById('chart-dca');
            const overlayEl = document.getElementById('chart-overlay');
            gdcaChart.applyOptions({{ width: gdcaEl.clientWidth, height: gdcaEl.clientHeight }});
            dcaChart.applyOptions({{ width: dcaEl.clientWidth, height: dcaEl.clientHeight }});
            overlayChart.applyOptions({{ width: overlayEl.clientWidth, height: overlayEl.clientHeight }});
        }}).observe(document.body);
        
        <div class="footer">
            <div class="brand">
                <span class="project-name">M Y C</span>
                <span class="divider"></span>
                <span class="creator">G R A P H</span>
            </div>
            <div class="system-info">
                Enterprise Edition Ready • Powered by Nautilus Trader
            </div>
        </div>
    </div>
    
    <style>
        .footer {
            margin-top: 60px;
            text-align: center;
            border-top: 1px solid rgba(255, 255, 255, 0.05);
            padding-top: 24px;
            margin-bottom: 24px;
        }

        .brand {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 16px;
            margin-bottom: 8px;
            font-weight: 700;
            letter-spacing: 2px;
        }

        .project-name {
            background: linear-gradient(90deg, #fff, #9ca3af);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 13px;
        }

        .creator {
            color: #4b5563;
            font-size: 11px;
        }

        .divider {
            width: 3px;
            height: 3px;
            background: #4b5563;
            border-radius: 50%;
        }

        .system-info {
            color: #4b5563;
            font-size: 10px;
            letter-spacing: 0.5px;
        }
    </style>
</body>
</html>'''

    # ========================================
    # GENERATE ALL HTML FILES
    # ========================================
    
    # GDCA Metrics for template
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
    
    # DCA Metrics for template
    dca_metrics = {
        'std_dca_invested': std_dca_invested,
        'std_dca_equity': std_dca_current_value,
        'std_dca_profit': std_dca_net_profit,
        'std_dca_roi': std_dca_roi,
        'std_dca_max_drawdown': std_dca_max_dd_pct
    }
    
    # Generate GDCA HTML
    gdca_html = generate_strategy_html(
        strategy_name="GDCA Strategy",
        strategy_color="#3b82f6",
        metrics=gdca_metrics,
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
        is_gdca=True
    )
    
    with open('gdca_result.html', 'w', encoding='utf-8') as f:
        f.write(gdca_html)
    print("Generated: gdca_result.html")
    
    # Generate DCA HTML
    dca_html = generate_strategy_html(
        strategy_name="Standard DCA",
        strategy_color="#8b5cf6",
        metrics=dca_metrics,
        json_ohlc=json_ohlc,
        json_equity=json_std_dca_equity,
        json_bnh=json_bnh,
        json_cash="[]",
        json_holdings="[]",
        json_markers="[]",
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
        is_gdca=False
    )
    
    with open('dca_result.html', 'w', encoding='utf-8') as f:
        f.write(dca_html)
    print("Generated: dca_result.html")
    
    # Generate Comparison HTML
    comparison_html = generate_comparison_html(
        gdca_metrics=gdca_metrics,
        dca_metrics=dca_metrics,
        json_equity_gdca=json_equity,
        json_equity_dca=json_std_dca_equity,
        json_bnh=json_bnh
    )
    
    with open('comparison_result.html', 'w', encoding='utf-8') as f:
        f.write(comparison_html)
    print("Generated: comparison_result.html")





















 

















 


 




















        



            






        





if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
