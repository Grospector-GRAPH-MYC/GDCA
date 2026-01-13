"""
Templates Module

Provides Jinja2 template rendering capabilities for HTML report generation.
"""

from .renderer import (
    render_template,
    save_html,
    render_and_save,
    render_strategy,
    render_comparison,
    render_index,
)

from .report_generator import generate_all_reports

__all__ = [
    'render_template',
    'save_html',
    'render_and_save',
    'render_strategy',
    'render_comparison',
    'render_index',
    'generate_all_reports',
]
