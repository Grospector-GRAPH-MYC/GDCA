import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

from nautilus_trader.model.data import Bar, BarType, BarSpecification, TradeTick, QuoteTick
from nautilus_trader.model.enums import BarAggregation, PriceType
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.persistence.catalog import ParquetDataCatalog
from nautilus_trader.core.datetime import dt_to_unix_nanos

import sys
# Add project root to path so we can import config
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import settings

def download_and_catalog(symbol_str="BTC/USD", venue_str="BITSTAMP", timeframe="1d", start_date="2011-01-01", data_type="bar"):
    exchange = ccxt.bitstamp()
    
    # Parse symbol
    base, quote = symbol_str.split("/")
    symbol = Symbol(f"{base}{quote}")
    venue = Venue(venue_str)
    instrument_id = InstrumentId(symbol, venue)
    
    # Parse Start Date
    since_ts = exchange.parse8601(f"{start_date}T00:00:00Z")
    print(f"Downloading {symbol_str} from {venue_str} since {start_date} ({timeframe})...")
    
    all_data = []
    
    if data_type == "bar":
        print(f"Downloading BARS {symbol_str} from {venue_str} since {start_date} ({timeframe})...")
        while True:
            try:
                ohlcv = exchange.fetch_ohlcv(symbol_str, timeframe, since=since_ts, limit=1000)
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                print(f"Fetched {len(ohlcv)} candles. Total: {len(all_data)}. Last: {pd.to_datetime(ohlcv[-1][0], unit='ms')}")
                
                # Update since_ts to last timestamp + 1 interval
                last_ts = ohlcv[-1][0]
                since_ts = last_ts + 1
                
                # Rate limit
                time.sleep(0.5)

            except Exception as e:
                print(f"Error fetching candles: {e}")
                break

    elif data_type == "trade":
        print(f"Downloading TRADES {symbol_str} from {venue_str} since {start_date}...")
        while True:
             try:
                 trades = exchange.fetch_trades(symbol_str, since=since_ts, limit=1000)
                 if not trades:
                     break
                 
                 all_data.extend(trades)
                 print(f"Fetched {len(trades)} trades. Total: {len(all_data)}. Last: {trades[-1]['datetime']}")
                 
                 # Update timestamp (ms)
                 last_ts = trades[-1]['timestamp']
                 # Create a delta to avoid duplicates if exchange supports pagination by ID usually, but timestamp is generic
                 if last_ts == since_ts:
                      since_ts += 1 # Force move forward
                 else:
                      since_ts = last_ts
                 
             except Exception as e:
                 print(f"Error fetching trades: {e}")
                 break
    
    if not all_data:
        print("No data downloaded.")
        return
            
    data_objects = []
    
    if data_type == "bar":
        # ... Bar Processing ...
        
        agg_map = {
            '1s': BarAggregation.SECOND,
            '1m': BarAggregation.MINUTE,
            '1h': BarAggregation.HOUR,
            '1d': BarAggregation.DAY,
        }
        
        bar_aggregation = agg_map.get(timeframe, BarAggregation.MINUTE)
        
        bar_spec = BarSpecification(
            step=1,
            aggregation=bar_aggregation,
            price_type=PriceType.LAST,
        )
        bar_type = BarType(instrument_id, bar_spec)
        
        print("Processing bars...")
        for row in all_data:
            ts, open_, high, low, close, volume = row
            # ... (Rest of bar processing) ...
            ts_ns = ts * 1_000_000
            bar = Bar(
                bar_type=bar_type,
                open=Price.from_str("{:.2f}".format(open_)),
                high=Price.from_str("{:.2f}".format(high)),
                low=Price.from_str("{:.2f}".format(low)),
                close=Price.from_str("{:.2f}".format(close)),
                volume=Quantity.from_str(str(volume)),
                ts_event=ts_ns,
                ts_init=ts_ns, 
            )
            data_objects.append(bar)

    elif data_type == "trade":
        print("Processing trades...")
        # Sort by timestamp
        all_data.sort(key=lambda x: x['timestamp'])
        
        for trade in all_data:
            ts_ns = trade['timestamp'] * 1_000_000
            price_str = "{:.2f}".format(trade['price'])
            qty_str = "{:.8f}".format(trade['amount'])
            
            # Nautilus TradeTick
            tick = TradeTick(
                instrument_id=instrument_id,
                price=Price.from_str(price_str),
                quantity=Quantity.from_str(qty_str),
                aggregator_id=None,
                ts_event=ts_ns,
                ts_init=ts_ns
            )
            data_objects.append(tick)
            
    print(f"Total processed {len(data_objects)} objects.")
    
    # Initialize Catalog
    catalog_path = Path("catalog")
    catalog = ParquetDataCatalog(catalog_path)
    
    print("Writing to catalog...")
    catalog.write_data(data_objects)
    print("Done.")

if __name__ == "__main__":
    # Example Usage:
    # download_and_catalog(data_type="bar", timeframe="1d")
    # download_and_catalog(data_type="trade", start_date="2023-01-01")
    
    # CI/CD / Env Var Configuration via Settings
    # Run with configuration from settings
    download_and_catalog(
        symbol_str=settings.DATA_SYMBOL,
        venue_str=settings.DATA_VENUE,
        timeframe=settings.DATA_TIMEFRAME,
        start_date=settings.DATA_START_DATE,
        data_type=settings.DATA_TYPE
    )
