"""Credible-set colocalisation: the `colocalisation` output, written by gentropy.

Descriptions are copied verbatim from croissant's `colocalisation.json`, as are its `isPrimaryKey`
and `foreign_key` tags (as `PrimaryKey`/`Reference`).

Rules confirmed against the 26.12 release (310,901,029 rows in 200 parts):

* Every row is one pair of credible sets scored by both methods at once: `colocalisationMethod` has
  the single value `COLOC_PIP_ECAVIAR`, and `h3`, `h4` and `clpp` are never null. So the pair is
  the primary key, as croissant says, with no method in it.
* The pair is not ordered (`leftLocusId > rightLocusId` in 83% of rows), and a set never pairs with
  itself.
* Both ids resolve to `credible_set.id`. The left set is always a GWAS one; `rightStudyType` is the
  `studyType` of the right set.
* `betaRatioSignAverage` is null in 18% of rows, and otherwise within [-1, 1].
* Chromosomes are 1-22 and X: no Y or MT pair was found.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, Note, PrimaryKey, Reference, RowCheck

#: Rounding slack allowed on a sum of probabilities.
_TOLERANCE = 1e-9

#: md5 hex digest: the shape of gentropy's credible set identifiers.
MD5_HEX = r'^[0-9a-f]{32}$'

#: `pts.schemas.study.StudyType`, in its order (a test keeps the two in sync).
StudyType = Literal['gwas', 'eqtl', 'pqtl', 'sqtl', 'tuqtl', 'sceqtl', 'scpqtl', 'scsqtl', 'sctuqtl']

Chromosome = Literal[
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT',
]  # fmt: skip


class Colocalisation(DatasetModel):
    """Colocalisation of pairs of overlapping credible sets."""

    dataset_name: ClassVar[str] = 'colocalisation'
    dataset_title: ClassVar[str] = 'Colocalisation with colocPIP and eCAVIAR'
    dataset_description: ClassVar[str] = (
        'GWAS-GWAS and GWAS-molQTL credible set colocalisation results using colocPIP and eCAVIAR methods. Dataset '
        'includes colocalising pairs and the statistics used to estimate the colocalisation'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'leftLocusId': '0a1b2c3d4e5f60718293a4b5c6d7e8f9',
        'rightLocusId': 'f9e8d7c6b5a4039281706f5e4d3c2b1a',
        'rightStudyType': 'eqtl',
        'chromosome': '1',
        'colocalisationMethod': 'COLOC_PIP_ECAVIAR',
        'numberColocalisingVariants': 12,
        'h3': 0.02,
        'h4': 0.97,
        'clpp': 0.41,
        'betaRatioSignAverage': 1.0,
    }
    row_checks: ClassVar[tuple[RowCheck, ...]] = (
        RowCheck(
            'h3_and_h4_are_exclusive',
            '`h3 + h4` is at most 1: they are posterior probabilities of exclusive hypotheses.',
            pl.col('h3') + pl.col('h4') <= 1.0 + _TOLERANCE,
        ),
    )

    leftLocusId: Annotated[
        str,
        PrimaryKey(),
        Reference('credible_set', 'id'),
        Note('always a GWAS credible set'),
    ] = Field(pattern=MD5_HEX, description='Identifier of the credible set of the left-side overlapping signal')
    rightLocusId: Annotated[str, PrimaryKey(), Reference('credible_set', 'id')] = Field(
        pattern=MD5_HEX,
        description='Identifier of the credible set of the right-side overlapping signal',
    )
    rightStudyType: Annotated[StudyType, Note('the `studyType` of the right-side credible set')] = Field(
        description='Type of the right-side study (e.g., gwas, eqtl)'
    )
    chromosome: Chromosome = Field(description='Chromosome where the colocalisation occurs')
    colocalisationMethod: Annotated[
        Literal['COLOC_PIP_ECAVIAR'],
        Note('both methods are run on every pair, and reported in one row'),
    ] = Field(description='Method used to estimate colocalisation (e.g., coloc)')
    numberColocalisingVariants: int = Field(
        ge=1,
        description='Number of variants shared by the two colocalising credible sets',
    )
    h3: float = Field(
        ge=0.0,
        le=1.0,
        description='Posterior probability that both traits are associated, but with different causal variants (H3)',
    )
    h4: float = Field(
        ge=0.0,
        le=1.0,
        description='Posterior probability that both traits are associated and share a causal variant (H4)',
    )
    clpp: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Colocalisation posterior probability (CLPP) score estimating the probability of shared causal variants'
        ),
    )
    betaRatioSignAverage: float | None = Field(
        ge=-1.0,
        le=1.0,
        description='Average sign of the beta ratio between colocalised variants',
    )
