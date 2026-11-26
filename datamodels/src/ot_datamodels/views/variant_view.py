"""Variant view: the `variant_view` view (pts `variant_view`), served as `variants`.

Every variant row with the ids of the credible sets it is in and of the pharmacogenomics records naming it.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import ClickHouseType
from ot_datamodels.outputs.patterns import MD5_HEX
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.serving import Serving

# list elements, so not `from_output`; tests/test_views.py checks them against credible_set.id and
# pharmacogenomics.pharmacogenomicsId
_CredibleSetId = Annotated[str, Field(pattern=MD5_HEX)]
_PharmacogenomicsId = Annotated[int, ClickHouseType('UInt64'), Field(ge=0)]


class VariantView(Variant):
    """Variant view."""

    dataset_name: ClassVar[str] = 'variant_view'
    dataset_title: ClassVar[str] = 'Variant view'
    dataset_description: ClassVar[str] = (
        'Variants with the identifiers of their credible sets and pharmacogenomics records nested in'
    )
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = Variant.example | {
        'credibleSetIds': ['0002c80bb07a1c4ab11e5b6d2c9e0d6c'],
        'pharmacogenomicsIds': [1],
    }

    credibleSetIds: list[_CredibleSetId] = Field(description='Credible sets the variant is in, sorted')
    pharmacogenomicsIds: list[_PharmacogenomicsId] = Field(
        description='pharmacogenomicsId of the pharmacogenomics records naming the variant, ascending'
    )
