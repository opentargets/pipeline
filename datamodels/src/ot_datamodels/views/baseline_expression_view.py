"""Baseline expression view: the `baseline_expression_view` view (pts `copy_view`).

A copy of `baseline_expression`, served from ClickHouse.
"""

from typing import ClassVar

from ot_datamodels.outputs.baseline_expression import BaselineExpression
from ot_datamodels.serving import Serving


class BaselineExpressionView(BaselineExpression):
    """Baseline expression view."""

    dataset_name: ClassVar[str] = 'baseline_expression_view'
    dataset_title: ClassVar[str] = 'Baseline expression view'
    dataset_description: ClassVar[str] = BaselineExpression.dataset_description
    clickhouse: ClassVar[Serving] = Serving(
        engine='mergetree',
        order_by=('targetId', 'tissueBiosampleId', 'celltypeBiosampleId'),
        column_types={'targetId': 'LowCardinality(String)'},
    )
