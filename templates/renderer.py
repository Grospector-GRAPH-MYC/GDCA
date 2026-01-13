"""
Template Renderer Module

Provides utilities for rendering Jinja2 templates with the shared base template.
"""

from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape

# Template directory
TEMPLATE_DIR = Path(__file__).parent

# Initialize Jinja2 environment with template inheritance support
_env = Environment(
    loader=FileSystemLoader(TEMPLATE_DIR),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)


def render_template(template_name: str, **context) -> str:
    """
    Render a Jinja2 template with the given context.
    
    Args:
        template_name: Name of the template file (e.g., 'strategy.html')
        **context: Template variables to pass to the template
        
    Returns:
        Rendered HTML string
    """
    template = _env.get_template(template_name)
    return template.render(**context)


def save_html(html_content: str, output_path: str) -> None:
    """
    Save rendered HTML content to a file.
    
    Args:
        html_content: The rendered HTML string
        output_path: Path to save the HTML file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def render_and_save(template_name: str, output_path: str, **context) -> None:
    """
    Render a template and save it to a file in one step.
    
    Args:
        template_name: Name of the template file
        output_path: Path to save the rendered HTML
        **context: Template variables
    """
    html = render_template(template_name, **context)
    save_html(html, output_path)


# Convenience functions for specific templates
def render_strategy(
    strategy_name: str,
    strategy_color: str,
    is_gdca: bool,
    invested: float,
    value: float,
    profit: float,
    roi: float,
    max_dd: float,
    btc_held: float,
    trades,
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
    backtest_start: str = None,
    backtest_end: str = None,
) -> str:
    """Render the strategy template with all required data."""
    profit_class = 'text-success' if profit >= 0 else 'text-danger'
    page_id = 'gdca' if is_gdca else 'dca'
    
    # Build date range text for header
    date_range_text = ""
    if backtest_start and backtest_end:
        date_range_text = f"{backtest_start} to {backtest_end}"
    elif backtest_start:
        date_range_text = f"From {backtest_start}"
    elif backtest_end:
        date_range_text = f"Until {backtest_end}"
    
    return render_template(
        'strategy.html',
        page_id=page_id,
        strategy_name=strategy_name,
        strategy_color=strategy_color,
        is_gdca=is_gdca,
        invested=invested,
        value=value,
        profit=profit,
        roi=roi,
        max_dd=max_dd,
        btc_held=btc_held,
        trades=trades,
        profit_class=profit_class,
        date_range_text=date_range_text,
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
    )


def render_comparison(
    gdca_metrics: dict,
    dca_metrics: dict,
    json_equity_gdca: str,
    json_equity_dca: str,
    json_bnh: str,
    backtest_start: str = None,
    backtest_end: str = None,
) -> str:
    """Render the comparison template."""
    # Create wrapper objects for template attribute access
    class MetricsWrapper:
        def __init__(self, data: dict):
            for key, val in data.items():
                setattr(self, key, val)
    
    gdca_wrapper = MetricsWrapper(gdca_metrics)
    dca_wrapper = MetricsWrapper(dca_metrics)
    gdca_wins = gdca_metrics.get('roi', 0) > dca_metrics.get('std_dca_roi', 0)
    
    # Build date range text for header
    date_range_text = ""
    if backtest_start and backtest_end:
        date_range_text = f"{backtest_start} to {backtest_end}"
    elif backtest_start:
        date_range_text = f"From {backtest_start}"
    elif backtest_end:
        date_range_text = f"Until {backtest_end}"
    
    return render_template(
        'comparison.html',
        page_id='comparison',
        gdca_metrics=gdca_wrapper,
        dca_metrics=dca_wrapper,
        gdca_wins=gdca_wins,
        date_range_text=date_range_text,
        json_equity_gdca=json_equity_gdca,
        json_equity_dca=json_equity_dca,
        json_bnh=json_bnh,
    )


def render_index(
    gdca_metrics: dict,
    dca_metrics: dict,
    backtest_start: str = None,
    backtest_end: str = None,
) -> str:
    """Render the index/dashboard template."""
    class MetricsWrapper:
        def __init__(self, data: dict):
            for key, val in data.items():
                setattr(self, key, val)
    
    # Build date range text for header
    date_range_text = ""
    if backtest_start and backtest_end:
        date_range_text = f"{backtest_start} to {backtest_end}"
    elif backtest_start:
        date_range_text = f"From {backtest_start}"
    elif backtest_end:
        date_range_text = f"Until {backtest_end}"
    
    return render_template(
        'index.html',
        page_id='index',
        gdca_metrics=MetricsWrapper(gdca_metrics),
        dca_metrics=MetricsWrapper(dca_metrics),
        date_range_text=date_range_text,
    )
