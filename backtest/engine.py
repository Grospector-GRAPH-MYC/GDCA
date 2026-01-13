"""
Backtest Engine Setup
=====================

This module handles the creation and configuration of the Nautilus Trader
BacktestEngine, including venue and instrument setup.
"""

from decimal import Decimal
from pathlib import Path

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.enums import AccountType, OmsType
from nautilus_trader.model.identifiers import InstrumentId, Venue, Symbol
from nautilus_trader.model.instruments import CurrencyPair
from nautilus_trader.model.objects import Money, Currency, Price, Quantity


def create_engine(log_level: str = "INFO") -> BacktestEngine:
    """
    Create and configure a BacktestEngine instance.
    
    Args:
        log_level: Logging level for the engine. Defaults to "INFO".
        
    Returns:
        Configured BacktestEngine instance.
    """
    engine_config = BacktestEngineConfig(
        logging=LoggingConfig(log_level=log_level),
    )
    return BacktestEngine(config=engine_config)


def create_instrument(
    instrument_id: InstrumentId,
    raw_symbol: str,
    base_currency: str = "BTC",
    quote_currency: str = "USD",
    price_precision: int = 2,
    size_precision: int = 8,
    maker_fee: Decimal = Decimal("0.001"),
    taker_fee: Decimal = Decimal("0.001"),
) -> CurrencyPair:
    """
    Create a CurrencyPair instrument for backtesting.
    
    Args:
        instrument_id: The instrument identifier.
        raw_symbol: The raw symbol string (e.g., "BTC/USD").
        base_currency: Base currency code. Defaults to "BTC".
        quote_currency: Quote currency code. Defaults to "USD".
        price_precision: Decimal places for price. Defaults to 2.
        size_precision: Decimal places for size. Defaults to 8.
        maker_fee: Maker fee rate. Defaults to 0.1%.
        taker_fee: Taker fee rate. Defaults to 0.1%.
        
    Returns:
        Configured CurrencyPair instrument.
    """
    currency_base = Currency.from_str(base_currency)
    currency_quote = Currency.from_str(quote_currency)
    
    price_increment = f"0.{'0' * (price_precision - 1)}1" if price_precision > 0 else "1"
    size_increment = f"0.{'0' * (size_precision - 1)}1" if size_precision > 0 else "1"
    
    return CurrencyPair(
        instrument_id=instrument_id,
        raw_symbol=Symbol(raw_symbol),
        base_currency=currency_base,
        quote_currency=currency_quote,
        price_precision=price_precision,
        size_precision=size_precision,
        price_increment=Price.from_str(price_increment),
        size_increment=Quantity.from_str(size_increment),
        lot_size=None,
        maker_fee=maker_fee,
        taker_fee=taker_fee,
        ts_event=0,
        ts_init=0,
    )


def add_venue(
    engine: BacktestEngine,
    venue: Venue,
    quote_currency: str = "USD",
    starting_balance: float = 1_000_000_000,
    oms_type: OmsType = OmsType.NETTING,
    account_type: AccountType = AccountType.MARGIN,
) -> None:
    """
    Add a venue to the backtest engine.
    
    Args:
        engine: The BacktestEngine instance.
        venue: The venue to add.
        quote_currency: Quote currency for the account. Defaults to "USD".
        starting_balance: Initial balance. Defaults to 1 billion.
        oms_type: Order management system type. Defaults to NETTING.
        account_type: Account type. Defaults to MARGIN.
    """
    currency = Currency.from_str(quote_currency)
    
    engine.add_venue(
        venue=venue,
        oms_type=oms_type,
        account_type=account_type,
        base_currency=currency,
        starting_balances=[Money(starting_balance, currency)]
    )


def normalize_symbol(symbol: str) -> str:
    """
    Normalize a symbol string for Nautilus ID creation.
    
    Removes slashes and other special characters that may cause issues.
    
    Args:
        symbol: Raw symbol string (e.g., "BTC/USD").
        
    Returns:
        Normalized symbol string (e.g., "BTCUSD").
    """
    return symbol.replace("/", "")
