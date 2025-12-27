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
        timeframe=settings.TIMEFRAME
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
    bars = catalog.bars([bar_type])
    print(f"Loaded {len(bars)} bars.")
    
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

    for current_time, row in df_bars.iterrows():
        # Daily Deposit Logic (DCA Plan)
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
        # Buy simply every day at Close price
        # Use the same daily amount as the strategy's DCA_AMOUNT
        daily_dca_amt = dca_sim_amount 
        current_close_price = row['Close']
        
        if current_close_price > 0:
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
    
    # HTML Template
    html_template = f"""
<!DOCTYPE html>
<html>
<head>
    <title>GDCA Backtest (Lightweight Charts)</title>
    <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
    <script>
        window.onerror = function(msg, url, lineNo, columnNo, error) {{
            var container = document.getElementById('container');
            if(container) {{
                container.innerHTML += '<div style="color:red; background:rgba(0,0,0,0.8); padding:20px;">' + 
                    '<h3>JavaScript Error:</h3>' +
                    '<p>' + msg + '</p>' +
                    '<p>Line: ' + lineNo + '</p>' +
                    '</div>';
            }}
            return false;
        }};
    </script>
    <style>
        body {{ margin: 0; padding: 0; background-color: #131722; color: #d1d4dc; font-family: 'Trebuchet MS', sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}
        #container {{ display: flex; flex-direction: column; height: 100%; width: 100%; position: relative; }}
        #chart-price {{ flex: 2; position: relative; width: 100%; }}
        #chart-equity {{ flex: 1; border-top: 1px solid #2a2e39; position: relative; width: 100%; }}
        
        .overlay-stats {{
            position: absolute;
            top: 20px;
            left: 20px;
            z-index: 20;
            background: rgba(30, 34, 45, 0.85);
            padding: 15px;
            border-radius: 6px;
            border: 1px solid #2a2e39;
            backdrop-filter: blur(4px);
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        .overlay-stats h2 {{ margin: 0 0 10px 0; font-size: 16px; color: #8C9FAD; }}
        .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; font-size: 14px; min-width: 350px; }}
        .stat-label {{ color: #787b86; }}
        .stat-value {{ font-weight: bold; color: #d1d4dc; }}
        .stat-value.green {{ color: #4caf50; }}
        .stat-value.red {{ color: #f44336; }}
        
        /* Log Scale Button */
        .controls {{
            position: absolute;
            top: 20px;
            right: 100px; /* Left of the price scale */
            z-index: 20;
        }}
        .btn {{
            background: #2962FF;
            color: white;
            border: none;
            padding: 6px 12px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 13px;
            opacity: 0.9;
        }}
        .btn:hover {{ opacity: 1; }}

        /* Custom Legend */
        .chart-legend {{
            position: absolute;
            left: 12px;
            top: 12px;
            z-index: 10;
            font-size: 12px;
            font-family: sans-serif;
            line-height: 18px;
            font-weight: 300;
            color: #fff;
            pointer-events: none; /* Click through */
            background: rgba(30, 34, 45, 0.85);
            padding: 8px 12px;
            border-radius: 4px;
            border: 1px solid #2a2e39;
            backdrop-filter: blur(4px);
        }}
        
    </style>
</head>
<body>
    <div id="container">
        <!-- Floating Stats -->
        <div class="overlay-stats">
            <h2>DCA Strategy Performance</h2>
            <div class="stat-row">
                <span class="stat-label">Total Invested:</span>
                <span class="stat-value">${metrics_payload['total_invested']:,.2f}</span>
            </div>
             <div class="stat-row">
                <span class="stat-label">BTC Acquired:</span>
                <span class="stat-value">{metrics_payload['total_btc']:.8f} BTC</span>
            </div>
             <div class="stat-row">
                <span class="stat-label">Current Value:</span>
                <span class="stat-value">${metrics_payload['total_equity']:,.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Net Profit:</span>
                <span class="stat-value { 'green' if metrics_payload['net_profit'] >= 0 else 'red' }">${metrics_payload['net_profit']:,.2f} ({metrics_payload['roi']:+.2f}%)</span>
            </div>
            
            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #2a2e39;"></div>
            
            <div class="stat-row">
                <span class="stat-label">Max Drawdown:</span>
                <span class="stat-value red">{metrics_payload['max_drawdown']:.2f}%</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Est. Trades:</span>
                <span class="stat-value">{metrics_payload['total_trades']}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Win Rate:</span>
                <span class="stat-value">{metrics_payload['win_rate']:.2f}%</span>
            </div>

            <div style="margin-top: 10px; padding-top: 10px; border-top: 1px solid #2a2e39;"></div>
            <h2>Standard DCA Performance</h2>
            <div class="stat-row">
                <span class="stat-label">Total Invested:</span>
                <span class="stat-value">${metrics_payload['std_dca_invested']:,.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Current Value:</span>
                <span class="stat-value">${metrics_payload['std_dca_equity']:,.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Net Profit:</span>
                <span class="stat-value { 'green' if metrics_payload['std_dca_profit'] >= 0 else 'red' }">${metrics_payload['std_dca_profit']:,.2f} ({metrics_payload['std_dca_roi']:+.2f}%)</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Max Drawdown:</span>
                <span class="stat-value red">{metrics_payload['std_dca_max_drawdown']:.2f}%</span>
            </div>
            
             <div style="margin-top: 15px; border-top: 1px solid #2a2e39; padding-top: 10px;">
                <span style="color: #787b86; font-size: 12px; display: block; margin-bottom: 5px;">Selected Order:</span>
                <div id="order-info" style="color: #64b5f6; font-weight: bold; min-height: 20px;">-</div>
            </div>
        </div>
        
        <!-- Controls -->
        <div class="controls">
             <button id="btn-log" class="btn">Log Scale: OFF</button>
             <button id="btn-reset" class="btn" style="background: #e0e0e0; color: #333; margin-left:10px;">Reset Zoom</button>
        </div>

        <div id="chart-price"></div>
        <div id="chart-equity"></div>
    </div>

    <script>
        // Data Injected from Python
        const ohlcData = {json_ohlc};
        const ema12Data = {json_ema12};
        const ema26Data = {json_ema26};
        
        // Marker Data
        const markerData = {json_markers};
        
        // Equity Data
        const equityData = {json_equity};
        const bnhData = {json_bnh};
        const cashData = {json_cash}; // New: Cash Data
        const holdingsData = {json_holdings}; // New: Holdings Data
        const stdDcaEquityData = {json_std_dca_equity}; // New: Standard DCA Equity Data
        
        // Zone Data
        const maShortData = {json_ma_short};
        const maStrongSellData = {json_ma_strong_sell};
        const maSellData = {json_ma_sell};
        const maBuyData = {json_ma_buy};
        const maStrongBuyData = {json_ma_strong_buy};
        const maLongData = {json_ma_long};
        
        // Ribbon Data (Python Calculated)
        const ribbons = {json_ribbons};
        const ribbonsPast = {json_ribbons_past};
        const futureData = {json_future};
        const pastData = {json_past};

        if (typeof LightweightCharts === 'undefined') {{
            document.getElementById('container').innerHTML = '<h2 style="color:red; text-align:center; padding:20px;">Error: Lightweight Charts library failed to load. Check internet connection or CDN URL.</h2>';
            console.error('LightweightCharts is undefined');
        }}

        console.log('OHLC Data Length:', ohlcData.length);
        console.log('Equity Data Length:', equityData.length);

        // --- Price Chart ---
        const priceContainer = document.getElementById('chart-price');
        const priceChart = LightweightCharts.createChart(priceContainer, {{
            layout: {{ background: {{ color: '#131722' }}, textColor: '#d1d4dc' }},
            grid: {{ vertLines: {{ color: '#2a2e39' }}, horzLines: {{ color: '#2a2e39' }} }},
            timeScale: {{ borderColor: '#2a2e39', timeVisible: true }},
            rightPriceScale: {{ borderColor: '#2a2e39', mode: LightweightCharts.PriceScaleMode.Logarithmic }}, // Default Log
        }});

        // GDCA Zone Lines (Ribbon & Boundaries)
        
        // Helper to Draw Pre-calculated Ribbon
        function drawRibbon(ribbonLists, boundaryData, colorHex, boundaryTitle, style) {{
            // Default style 0 (Solid)
            const lineStyle = style !== undefined ? style : 0;
            
            // Draw Intermediate Lines
            if (ribbonLists) {{
                for (const lineData of ribbonLists) {{
                    const lineSeries = priceChart.addLineSeries({{
                        color: colorHex, 
                        lineWidth: 1, 
                        crosshairMarkerVisible: false,
                        priceLineVisible: false,
                        lastValueVisible: false,
                        lineStyle: lineStyle
                    }});
                    lineSeries.setData(lineData);
                }}
            }}
            
             // Draw Boundary Lines (Top of this zone)
             // Only draw boundary if boundaryData is provided
             if (boundaryData) {{
                const lineSeries = priceChart.addLineSeries({{
                    color: colorHex, 
                    lineWidth: 2, 
                    title: '', // Hide title per request ("Don't show zone")
                    lastValueVisible: false, // Hide axis label
                    lineStyle: lineStyle,
                    crosshairMarkerVisible: false // Hide crosshair point too? Maybe keep default.
                }});
                lineSeries.setData(boundaryData);
                
                // Add Pointers (Markers) at Start and End per request
                if (boundaryData.length > 0) {{
                    const first = boundaryData[0];
                    const last = boundaryData[boundaryData.length - 1];
                    lineSeries.setMarkers([
                        {{ time: first.time, position: 'inBar', color: colorHex, shape: 'circle', size: 1 }},
                        {{ time: last.time, position: 'inBar', color: colorHex, shape: 'circle', size: 1 }}
                    ]);
                }}
             }}
        }}

        // 1. Short Zone
        drawRibbon(ribbons.short, maShortData, '#801922', 'Short Zone');
        
        // 2. Strong Sell Zone
        drawRibbon(ribbons.strong_sell, maStrongSellData, '#b22833', 'Strong Sell');
        
        // 3. Sell Zone (Normal)
        drawRibbon(ribbons.sell, maSellData, '#f57f17', 'Sell (Normal)');

        // 4. Buy Zone
        drawRibbon(ribbons.buy, maBuyData, '#1b5e20', 'Buy');
        
        // 5. Long Zone
        drawRibbon(ribbons.long, maStrongBuyData, '#00332a', 'Strong Buy');
        
        // Bottom-most line (Long) - Draw manually as boundary
        const lineLong = priceChart.addLineSeries({{ 
            color: '#00332a', 
            lineWidth: 2, 
            title: '', 
            lastValueVisible: false,
            crosshairMarkerVisible: false
        }});
        lineLong.setData(maLongData);
        if (maLongData && maLongData.length > 0) {{
             lineLong.setMarkers([
                 {{ time: maLongData[0].time, position: 'inBar', color: '#00332a', shape: 'circle', size: 1 }},
                 {{ time: maLongData[maLongData.length - 1].time, position: 'inBar', color: '#00332a', shape: 'circle', size: 1 }}
             ]);
        }}
        
        // --- Draw Past Ribbons (Projected) ---
        if (ribbonsPast) {{
            const dashed = 2; // LineStyle.Dashed
            
            // Note: We don't have separate "Boundary Data" arrays for past, 
            // but the ribbon structure itself is all we need for visualization?
            // Actually, drawRibbon expects lists of lines.
            // Our ribbonsPast contains keys 'short', 'buy' etc. which are lists of lines.
            // We can just call drawRibbon with boundaryData=null.
            
            drawRibbon(ribbonsPast.short, null, '#801922', '', dashed);
            drawRibbon(ribbonsPast.strong_sell, null, '#b22833', '', dashed);
            drawRibbon(ribbonsPast.sell, null, '#f57f17', '', dashed);
            drawRibbon(ribbonsPast.buy, null, '#1b5e20', '', dashed);
            drawRibbon(ribbonsPast.long, null, '#00332a', '', dashed);
        }}
        
        // --- Future Dates Extension ---
        // Add an invisible series to force the timeline to extend
        if (futureData && futureData.length > 0) {{
            // Using a LineSeries with transparent color
            const futureSeries = priceChart.addLineSeries({{
                color: 'rgba(0,0,0,0)', 
                lineWidth: 0,
                crosshairMarkerVisible: false,
                priceLineVisible: false, 
                lastValueVisible: false
            }});
            futureSeries.setData(futureData);
        }}
        
        // --- Past Dates Extension ---
        if (pastData && pastData.length > 0) {{
             const pastSeries = priceChart.addLineSeries({{
                color: 'rgba(0,0,0,0)', 
                lineWidth: 0,
                crosshairMarkerVisible: false,
                priceLineVisible: false, 
                lastValueVisible: false
             }});
             pastSeries.setData(pastData);
        }}
        
        // --- EMA Lines (Restored) ---
        const line12 = priceChart.addLineSeries({{ color: '#f23645', lineWidth: 1, title: 'EMA 12' }});
        line12.setData(ema12Data);
        
        const line26 = priceChart.addLineSeries({{ color: '#2962FF', lineWidth: 1, title: 'EMA 26' }});
        line26.setData(ema26Data);

        // --- Candlesticks (Moved here to be ON TOP of ribbons) ---
        const candleSeries = priceChart.addCandlestickSeries({{
            upColor: '#089981', downColor: '#f23645', borderVisible: false, wickUpColor: '#089981', wickDownColor: '#f23645'
        }});
        candleSeries.setData(ohlcData);
        candleSeries.setMarkers(markerData);


        // --- Equity Chart ---
        const equityChart = LightweightCharts.createChart(document.getElementById('chart-equity'), {{
            width: document.getElementById('chart-equity').clientWidth,
            height: 500,
            layout: {{
                backgroundColor: '#131722',
                textColor: '#d1d4dc',
            }},
            grid: {{
                vertLines: {{ color: '#2B2B43', style: 1, visible: true }},
                horzLines: {{ color: '#2B2B43', style: 1, visible: true }},
            }},
            rightPriceScale: {{
                borderColor: 'rgba(197, 203, 206, 0.8)',
            }},
            timeScale: {{
                borderColor: 'rgba(197, 203, 206, 0.8)',
            }},
        }});
        
        // Resize handler
        window.addEventListener('resize', () => {{
            equityChart.resize(document.getElementById('chart-equity').clientWidth, 500);
        }});

        // --- GDCA SERIES ---

        // Portfolio Value
        const equitySeries = equityChart.addAreaSeries({{
            topColor: 'rgba(41, 98, 255, 0.3)', 
            bottomColor: 'rgba(41, 98, 255, 0)', 
            lineColor: '#2962FF', 
            lineWidth: 2,
            title: 'GDCA Portfolio Value'
        }});
        equitySeries.setData(equityData);
        
        // Cash Series
        const cashSeries = equityChart.addAreaSeries({{
            topColor: 'rgba(76, 175, 80, 0.5)', 
            bottomColor: 'rgba(76, 175, 80, 0.05)', 
            lineColor: '#4CAF50', 
            lineWidth: 1,
            title: 'Cash Holdings'
        }});
        cashSeries.setData(cashData);
        
        // Net Invested
        const bnhSeries = equityChart.addLineSeries({{
            color: '#FF9800', 
            lineWidth: 2,
            lineStyle: 2, 
            title: 'Net Invested'
        }});
        bnhSeries.setData(bnhData);

        // Holdings (Left Axis)
        const holdingsSeries = equityChart.addAreaSeries({{
            topColor: 'rgba(255, 235, 59, 0.3)', 
            bottomColor: 'rgba(255, 235, 59, 0.05)', 
            lineColor: '#FFEB3B', 
            lineWidth: 1,
            title: 'BTC Holdings',
            priceScaleId: 'left' // Use Left Scale
        }});
        holdingsSeries.setData(holdingsData);
        holdingsSeries.setData(holdingsData);
        
        // --- STANDARD DCA SERIES (Merged into Equity Chart) ---
        const stdDcaSeries = equityChart.addLineSeries({{
            color: '#9C27B0', // Purple
            lineWidth: 2,
            lineStyle: 0, // Solid
            title: 'Standard DCA Value'
        }});
        stdDcaSeries.setData(stdDcaEquityData);
        
        // Also plot Invested Capital on Std DCA chart for reference? 
        // Technically Std DCA Invested should be same as GDCA Invested if logic holds, 
        // but let's re-use the 'bnhData' (Invested Capital) for context or just leave it clean.
        // Let's add it for context so user sees the gain.
        // Removed redundant stdDcaInvestedSeries (bnhSeries covers it)

        // --- INDICATORS (On GDCA Chart) ---

        // Helper to add line
        function addLine(data, color, title, width=1) {{
            const s = equityChart.addLineSeries({{
                color: color,
                lineWidth: width,
                lineStyle: 0,
                title: title,
                crosshairMarkerVisible: false
            }});
            s.setData(data);
            return s;
        }}
        
        addLine(maStrongBuyData, '#00FF00', 'CDC Strong Buy');
        addLine(maBuyData, '#81C784', 'CDC Buy');
        addLine(maShortData, '#FFFF00', 'CDC Short');
        addLine(maLongData, '#ff00ff', 'CDC Long'); 
        addLine(maSellData, '#E57373', 'CDC Sell');
        addLine(maStrongSellData, '#FF0000', 'CDC Strong Sell');
        
        // Markers on Equity Series or separate? 
        // Markers usually go on the series they relate to. 
        // Since we buy/sell BTC, let's put them on the Portfolio Value series or create a phantom series if needed.
        equitySeries.setMarkers(markerData);
        
        // Fit Content
        equityChart.timeScale().fitContent();

        // Sync Timescales (Optional but good)
        // Simple one-way sync attempt or just let them be independent
        // Keeping independent for simplicity as requested.
        
        // --- Sync Charts & Resize ---
        function syncVisibleRange(source, target) {{
            const visibleRange = source.timeScale().getVisibleRange();
            if (visibleRange) {{
                target.timeScale().setVisibleRange(visibleRange);
            }}
        }}

        priceChart.timeScale().subscribeVisibleTimeRangeChange(() => syncVisibleRange(priceChart, equityChart));
        equityChart.timeScale().subscribeVisibleTimeRangeChange(() => syncVisibleRange(equityChart, priceChart));


        new ResizeObserver(entries => {{
             for (let entry of entries) {{
                 const {{ width, height }} = entry.contentRect;
                 if (entry.target.id === 'chart-price') priceChart.applyOptions({{ width, height }});
             }}
        }}).observe(priceContainer);
        
        // The equityContainer and stdDcaChart resize are handled by the window resize listener now.
        // Remove the old equityContainer observer.
        
        // --- Order Info Interaction ---
        const orderInfoEl = document.getElementById('order-info');
        
        // Create Map for O(1) lookup
        const markerMap = new Map();
        if (markerData) {{
            markerData.forEach(m => markerMap.set(m.time, m));
        }}
        
        priceChart.subscribeCrosshairMove(param => {{
            if (!param.time) {{
                orderInfoEl.innerHTML = '-';
                return;
            }}
            
            const marker = markerMap.get(param.time);
            if (marker) {{
                orderInfoEl.innerHTML = marker.tooltip;
                orderInfoEl.style.color = marker.color; // Use marker color (Red/Blue)
            }} else {{
                // Optional: Keep last separate, or clear. clearing is cleaner.
                orderInfoEl.innerHTML = '-';
                orderInfoEl.style.color = '#64b5f6'; // Reset to default
            }}
        }});
        
        // Auto-fit handled by zoomOutFull()

        // --- Log Scale Toggle ---
        const btnLog = document.getElementById('btn-log');
        let isLog = true; // Default ON
        
        // Initial Button State
        btnLog.innerText = 'Log Scale: ' + (isLog ? 'ON' : 'OFF');
        btnLog.style.background = isLog ? '#ff9800' : '#2962FF'; 
        
        btnLog.addEventListener('click', () => {{
            isLog = !isLog;
            const mode = isLog ? LightweightCharts.PriceScaleMode.Logarithmic : LightweightCharts.PriceScaleMode.Normal;
            priceChart.priceScale('right').applyOptions({{ mode: mode }});
            btnLog.innerText = 'Log Scale: ' + (isLog ? 'ON' : 'OFF');
            btnLog.style.background = isLog ? '#ff9800' : '#2962FF'; 
        }});

        // --- Reset Zoom / Zoom Out 100% ---
        const btnReset = document.getElementById('btn-reset');
        
        function zoomOutFull() {{
            // Calculate the absolute min and max time across all datasets
            let minTime = Infinity;
            let maxTime = -Infinity;
            
            const checkData = (data) => {{
                if (data && data.length > 0) {{
                    const first = data[0].time;
                    const last = data[data.length - 1].time;
                    if (first < minTime) minTime = first;
                    if (last > maxTime) maxTime = last;
                }}
            }};
            
            // Only fit to OHLC data (Trade Data), ignoring past/future invisible extensions
            checkData(ohlcData);
            
            if (minTime !== Infinity && maxTime !== -Infinity) {{
                // Apply a small buffer if needed, or set exact range
                priceChart.timeScale().setVisibleRange({{ from: minTime, to: maxTime }});
            }} else {{
                priceChart.timeScale().fitContent();
            }}
        }}

        btnReset.addEventListener('click', zoomOutFull);
        
        // Initial Zoom
        // Timeout to ensure chart is fully rendered before setting range
        setTimeout(zoomOutFull, 100);

    </script>
</body>
</html>
    """
    
    with open('backtest_result.html', 'w') as f:
        f.write(html_template)
    
    print(f"Visualization saved to backtest_result.html")


if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
