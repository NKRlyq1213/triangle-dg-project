from __future__ import annotations

from .table1_rules import load_table1_rule
from .table2_rules import load_table2_rule


def load_rule(table: str, order: int) -> dict:
    """
    Unified entry point for loading a quadrature rule.
    """
    table = table.lower().strip()
    if table == "table1":
        return load_table1_rule(order)
    if table == "table2":
        return load_table2_rule(order)
    raise ValueError("table must be either 'table1' or 'table2'.")