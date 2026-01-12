from decimal import Decimal
from dataclasses import dataclass
from typing import Optional, Dict

from nautilus_trader.config import StrategyConfig
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import BarAggregation, PriceType, OrderSide, TimeInForce, TriggerType
from nautilus_trader.model.identifiers import InstrumentId
from nautilus_trader.model.objects import Price, Quantity
from nautilus_trader.trading.strategy import Strategy
try:
    # New path (v1.221.0+)
    from nautilus_trader.indicators import SimpleMovingAverage
    from nautilus_trader.indicators import ExponentialMovingAverage
except ImportError:
    # Old path (pre-v1.221.0)
    from nautilus_trader.indicators.average.sma import SimpleMovingAverage 
    from nautilus_trader.indicators.average.ema import ExponentialMovingAverage

class GDCAConfig(StrategyConfig, frozen=True):
    instrument_id: str
    dca_amount: str = "1000.0"       # DCA Plan (USD)
    
    # MA Settings
    ma_period: int = 1460            # 365 * 4
    strong_ma_period: int = 365
    
    # Multipliers (Mode 1 Manual)
    manual_short_multi: float = 2.618
    manual_strong_sell_multi: float = 2.618
    manual_sell_multi: float = 3.618
    manual_buy_multi: float = 1.0
    manual_strong_buy_multi: float = 0.618
    manual_long_multi: float = 0.382
    
    # CDC Action Zone
    cdc_action_switch: bool = True
    cdc_fast: int = 12
    cdc_slow: int = 26
    
    # Zone Settings
    start_short_zone: int = 100
    end_short_zone: int = 0
    start_sell_zone: int = 100
    end_sell_zone: int = 0
    start_normal_zone: int = 0
    end_normal_zone: int = 100
    start_buy_zone: int = 100
    end_buy_zone: int = 200
    start_long_zone: int = 0
    end_long_zone: int = 100

    timeframe: str = "1d"
    use_reserve: bool = True
    use_derivative: bool = True
    
    # Compatibility
    lot_size: str = "0.001" 
    step_drop_pct: str = "0.01"
    take_profit_pct: str = "0.05"
    max_positions: int = 10
    
    # Backtest Execution Filter
    backtest_start: Optional[str] = None
    backtest_end: Optional[str] = None


class GDCAStrategy(Strategy):
    """
    Grospector DCA V.4 Port
    """
    @staticmethod
    def calculate_visualization_data(df, config: GDCAConfig = None):
        """
        Calculates indicators and zones on a Pandas DataFrame for visualization.
        """
        import pandas as pd
        import numpy as np
        
        # Use default config if none provided (for constants)
        # Or hardcode defaults matching Config class if cleaner
        if config is None:
            # We can't instantiate config easily without required fields, 
            # so let's rely on internal defaults or passed config.
            # Ideally Config should be passed.
            pass
            
        # Defaults (matching GDCAConfig)
        ma_period = config.ma_period if config else 1460
        strong_ma_period = config.strong_ma_period if config else 365
        
        m_short = config.manual_short_multi if config else 2.618
        m_ssell = config.manual_strong_sell_multi if config else 2.618
        m_sell = config.manual_sell_multi if config else 3.618
        m_buy = config.manual_buy_multi if config else 1.0
        m_sbuy = config.manual_strong_buy_multi if config else 0.618
        m_long = config.manual_long_multi if config else 0.382
        
        # 1. Base MA
        df['MA_Base'] = df['Close'].rolling(window=ma_period).mean()
        
        # 2. MA of MA
        df['MA_of_MA'] = df['MA_Base'].rolling(window=strong_ma_period).mean()
        
        # 3. CDC EMAs
        df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # Calculate Lines
        ma_buy = df['MA_Base'] * m_buy
        ma_strong_buy = df['MA_of_MA'] * m_buy * m_sbuy
        
        ma_sell = df['MA_Base'] * m_sell
        ma_strong_sell = df['MA_of_MA'] * m_sell * m_ssell
        
        ma_short = ma_strong_sell * m_short
        ma_long = ma_strong_buy * m_long
        
        # Store in DF
        df['MA_Buy'] = ma_buy
        df['MA_Strong_Buy'] = ma_strong_buy
        df['MA_Sell'] = ma_sell
        df['MA_Strong_Sell'] = ma_strong_sell
        df['MA_Short'] = ma_short
        df['MA_Long'] = ma_long

        # Channel Boundaries & Gaps (Optional, if needed by caller, or let caller derive)
        # But let's verify what caller needs.
        # Caller uses these for 'ribbons' drawing.
        
        return df

    def __init__(self, config: GDCAConfig):
        super().__init__(config=config)
        
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.dca_amount = Decimal(config.dca_amount)
        
        # State
        self.invested = Decimal("0.0")
        self.reserve = Decimal("0.0")
        
        # Indicators
        self.ma_base = SimpleMovingAverage(config.ma_period)
        self.ma_of_ma = SimpleMovingAverage(config.strong_ma_period)
        
        # CDC Indicators
        self.ema_fast = ExponentialMovingAverage(config.cdc_fast)
        self.ema_slow = ExponentialMovingAverage(config.cdc_slow)
        
        # Multipliers
        self.short_multi = Decimal(config.manual_short_multi)
        self.strong_sell_multi = Decimal(config.manual_strong_sell_multi)
        self.sell_multi = Decimal(config.manual_sell_multi)
        self.buy_multi = Decimal(config.manual_buy_multi)
        self.strong_buy_multi = Decimal(config.manual_strong_buy_multi)
        self.strong_buy_multi = Decimal(config.manual_strong_buy_multi)
        self.long_multi = Decimal(config.manual_long_multi)
        
        # Parse Backtest Date Filter
        import pandas as pd
        self.start_ts = pd.Timestamp(config.backtest_start).value if config.backtest_start else None
        self.end_ts = pd.Timestamp(config.backtest_end).value if config.backtest_end else None

    def on_start(self):
        self.log.info(f"GDCA V.4 Strategy started for {self.instrument_id}")
        
        agg_map = {
            '1m': BarAggregation.MINUTE,
            '1h': BarAggregation.HOUR,
            '1d': BarAggregation.DAY,
        }
        
        bar_type = BarType(
            self.instrument_id,
            BarSpecification(1, agg_map.get(self.config.timeframe, BarAggregation.DAY), PriceType.LAST)
        )
        self.subscribe_bars(bar_type)

    def on_bar(self, bar: Bar):
        # Update Indicators
        close_price = bar.close.as_double()
        self.ma_base.update_raw(close_price)
        
        # Helper to get current MA value if ready, else 0 or use bar.close to avoid None errors (logic check needed)
        if not self.ma_base.initialized:
            return # Warmup
            
        ma_base_val = self.ma_base.value
        
        # Update Recursive Indicator (SMA of SMA)
        # Pine: maStrongBuy = ta.sma(maBuy * strongBuyMulti, strongLen)
        # We actually need to feed the *unscaled* MA base into the second SMA to keep it generic,
        # then apply scalings at the end.
        self.ma_of_ma.update_raw(ma_base_val)
        
        # Update CDC
        self.ema_fast.update_raw(close_price)
        self.ema_slow.update_raw(close_price)
        
        if not self.ma_of_ma.initialized or not self.ema_slow.initialized:
            return
            
        # --- Backtest Date Filter Check ---
        # Allow indicators to calculate above, but stop execution if outside range
        if self.start_ts and bar.ts_init < self.start_ts:
            return
        if self.end_ts and bar.ts_init > self.end_ts:
            return
            
        # --- Calculate Lines ---
        price = bar.close.as_decimal()
        
        # 1. Buy Line (maBuy)
        ma_buy = Decimal(self.ma_base.value) * self.buy_multi
        
        # 2. Strong Buy Line (maStrongBuy) = SMA(SMA) * BuyMulti * StrongBuyMulti
        ma_strong_buy = Decimal(self.ma_of_ma.value) * self.buy_multi * self.strong_buy_multi
        
        # 3. Sell Line (maSell)
        ma_sell = Decimal(self.ma_base.value) * self.sell_multi
        
        # 4. Strong Sell Line (maStrongSell)
        ma_strong_sell = Decimal(self.ma_of_ma.value) * self.sell_multi * self.strong_sell_multi
        
        # 5. Short Line
        ma_short = ma_strong_sell * self.short_multi
        
        # 6. Long Line
        ma_long = ma_strong_buy * self.long_multi
        
        # --- Channel Boundaries ---
        higher_sell = max(ma_sell, ma_strong_sell)
        lower_sell = min(ma_sell, ma_strong_sell)
        
        higher_buy = max(ma_buy, ma_strong_buy)
        lower_buy = min(ma_buy, ma_strong_buy)
        
        # --- Channel Gaps ---
        gap_short = (ma_short - higher_sell) / 10
        gap_sell = (higher_sell - lower_sell) / 10
        gap_normal = (lower_sell - higher_buy) / 10
        gap_buy = (higher_buy - ma_strong_buy) / 10
        gap_long = (ma_strong_buy - ma_long) / 10
        
        # --- Determine Logic State ---
        is_short_zone = price > higher_sell
        is_sell_zone = (price > lower_sell) and (price < higher_sell)
        is_half_buy_zone = (price < lower_sell) and (price > higher_buy)
        is_buy_zone = (price < higher_buy) and (price > lower_buy)
        is_long_zone = price < lower_buy
        
        # --- Calculate Percent ---
        percent = Decimal(0)
        
        c_ss = self.config.start_short_zone
        c_es = self.config.end_short_zone
        step_short = abs(c_ss - c_es) / 10
        
        c_sell_s = self.config.start_sell_zone
        c_sell_e = self.config.end_sell_zone
        step_sell = abs(c_sell_s - c_sell_e) / 10
        
        c_norm_s = self.config.start_normal_zone
        c_norm_e = self.config.end_normal_zone
        step_norm = abs(c_norm_s - c_norm_e) / 10
        
        c_buy_s = self.config.start_buy_zone
        c_buy_e = self.config.end_buy_zone
        step_buy = abs(c_buy_s - c_buy_e) / 10
        
        c_long_s = self.config.start_long_zone
        c_long_e = self.config.end_long_zone
        step_long = abs(c_long_s - c_long_e) / 10

        if is_short_zone:
            found = False
            for i in range(1, 11):
                chan_val = ma_short - (gap_short * i)
                if price > chan_val:
                    percent = Decimal(c_ss - (step_short * i))
                    found = True
                    break
            if not found:
                 percent = Decimal(c_ss - (step_short * 10))
                 
        elif is_sell_zone:
            found = False
            for i in range(0, 10):
                chan_val = higher_sell - (gap_sell * (i+1))
                if price > chan_val:
                    percent = Decimal(c_sell_s - (step_sell * i))
                    found = True
                    break
            if not found:
                percent = Decimal(c_sell_s - (step_sell * 10))
                
        elif is_half_buy_zone:
            found = False
            for i in range(1, 11):
                chan_val = lower_sell - (gap_normal * i)
                if price > chan_val:
                    percent = Decimal(c_norm_s + (step_norm * i))
                    found = True
                    break
            if not found:
                percent = Decimal(c_norm_s + (step_norm * 10))

        elif is_buy_zone:
            found = False
            for i in range(1, 11):
                chan_val = higher_buy - (gap_buy * i)
                if price > chan_val:
                    percent = Decimal(c_buy_s + (step_buy * i))
                    found = True
                    break
            if not found:
                percent = Decimal(c_buy_s + (step_buy * 10))
                
        elif is_long_zone:
            found = False
            for i in range(1, 11):
                chan_val = ma_strong_buy - (gap_long * i)
                if price > chan_val:
                    percent = Decimal(c_long_s + (step_long * i))
                    found = True
                    break
            if not found:
                 percent = Decimal(c_long_s + (step_long * 10))

        # --- CDC Action Zone ---
        fast = Decimal(self.ema_fast.value)
        slow = Decimal(self.ema_slow.value)
        bull = fast > slow
        bear = fast < slow
        
        # --- Amount Calculation (Savings Plan Logic) ---
        # 1. Accumulate Savings (Virtual Cash)
        self.reserve += self.dca_amount
        
        # 2. Calculate Potential Order Value (USD)
        amount_usd = Decimal(0)
        
        if percent > 100:
             # Excess Logic: Base DCA + Excess % of Total Savings
             excess_pct = (percent - 100)
             # User Request: "10% * now cash" (if 110%)
             amount_usd = self.dca_amount + (self.reserve * excess_pct / 100)
        else:
             # Normal Logic: % of DCA Amount
             amount_usd = self.dca_amount * percent / 100
             
        # 3. Calculate Quantity
        # Note: quantity depends on current price
        entry_qty = amount_usd / price
        
        # --- Execution Signals ---
        should_buy = False
        
        if self.config.cdc_action_switch:
            if is_buy_zone and bull:
                should_buy = True
            elif is_half_buy_zone and bull:
                should_buy = True
        else:
            if is_buy_zone or is_half_buy_zone:
                should_buy = True

        should_short = False
        if self.config.use_derivative:
            # Pine: if(bar and short and cdcActionZoneBear) strategy.order(..., strategy.short, ...)
            if self.config.cdc_action_switch:
                 if is_short_zone and bear:
                     should_short = True
            else:
                 if is_short_zone:
                     should_short = True
                     
        # Long Zone Logic (Deep Buy)
        if self.config.use_derivative:
             if self.config.cdc_action_switch:
                  if is_long_zone and bull:
                       should_buy = True 
             else:
                  if is_long_zone:
                       should_buy = True
        
        # --- Execution & Fund Deduction ---
        # Only spend the money if we actually execute
        
        if should_buy and entry_qty > 0:
             # Ensure we don't hold a SHORT if we want to BUY (Netting/Reversal)
             # Nautilus does netting by default on OmsType.NETTING.
             
             qty_quant = Quantity(entry_qty, precision=8)
             if qty_quant.as_decimal() > 0:
                 self.buy(self.instrument_id, qty_quant)
                 # Deduct cost from Virtual Cash (Savings)
                 # We assume close execution = calculated price
                 self.reserve -= amount_usd
                 if self.reserve < 0: self.reserve = Decimal(0) # Safety
                 
        if should_short and entry_qty > 0:
             qty_quant = Quantity(entry_qty, precision=8)
             if qty_quant.as_decimal() > 0:
                 self.sell(self.instrument_id, qty_quant)
                 # Deduct margin from Virtual Cash
                 self.reserve -= amount_usd
                 if self.reserve < 0: self.reserve = Decimal(0)
        
        # --- Exit Logic ---
        # Close on Sell Zone & Bear
        if is_sell_zone and bear:
             self.close_all_positions(self.instrument_id)

    def buy(self, instrument_id: InstrumentId, quantity: Quantity):
        # Place Market Order
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.BUY,
            quantity=quantity,
        )
        self.submit_order(order)

    def sell(self, instrument_id: InstrumentId, quantity: Quantity):
        # Place Market Order (Short)
        order = self.order_factory.market(
            instrument_id=instrument_id,
            order_side=OrderSide.SELL,
            quantity=quantity,
        )
        self.submit_order(order)
