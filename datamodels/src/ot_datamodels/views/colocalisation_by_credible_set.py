"""Colocalisation by credible set: the `colocalisation_by_credible_set` view, served as `colocalisation`.

Written by pts `colocalisation_view`. One row per credible set with every colocalisation it takes part in, from
both sides: an output row as is (`leftLocusId` as `credibleSetId`) and swapped (`rightLocusId` as `credibleSetId`,
`rightStudyType` `gwas`, the left set's type). Repeated tuples are served once, sorted by their fields in order
(`otherCredibleSetId`, `rightStudyType` and `chromosome` in vocabulary order, `colocalisationMethod`, then the
statistics, nulls last).
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.colocalisation import Colocalisation
from ot_datamodels.vocabularies import ColocalisationMethod

CredibleSetId = from_output(Colocalisation, 'leftLocusId', primary_key=True)
RowCredibleSetId = from_output(Colocalisation, 'leftLocusId')
OtherCredibleSetId = from_output(Colocalisation, 'rightLocusId')
RightStudyType = from_output(Colocalisation, 'rightStudyType')
ColocChromosome = from_output(Colocalisation, 'chromosome')
NumberColocalisingVariants = from_output(Colocalisation, 'numberColocalisingVariants')
H3 = from_output(Colocalisation, 'h3')
H4 = from_output(Colocalisation, 'h4')
Clpp = from_output(Colocalisation, 'clpp')
BetaRatioSignAverage = from_output(Colocalisation, 'betaRatioSignAverage')

_EXAMPLE = Colocalisation.example
_METHOD = Colocalisation.model_fields['colocalisationMethod'].description


class ColocalisationRow(StructModel):
    """`colocalisation[]`: one colocalisation seen from `credibleSetId`, the method served as a string."""

    credibleSetId: RowCredibleSetId
    otherCredibleSetId: OtherCredibleSetId
    rightStudyType: RightStudyType
    chromosome: ColocChromosome
    colocalisationMethod: ColocalisationMethod = Field(description=_METHOD)
    numberColocalisingVariants: NumberColocalisingVariants
    h3: H3
    h4: H4
    clpp: Clpp
    betaRatioSignAverage: BetaRatioSignAverage


class ColocalisationByCredibleSet(DatasetModel):
    """Colocalisation by credible set."""

    dataset_name: ClassVar[str] = 'colocalisation_by_credible_set'
    dataset_title: ClassVar[str] = 'Colocalisation by credible set'
    dataset_description: ClassVar[str] = 'The colocalisations of each credible set, from either side'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'credibleSetId': _EXAMPLE['leftLocusId'],
        'colocalisation': [
            {
                'credibleSetId': _EXAMPLE['leftLocusId'],
                'otherCredibleSetId': _EXAMPLE['rightLocusId'],
                **{k: _EXAMPLE[k] for k in list(_EXAMPLE)[2:]},
            }
        ],
    }

    credibleSetId: CredibleSetId
    colocalisation: list[ColocalisationRow] = Field(
        min_length=1, description='Colocalisations of the credible set, with the other set from either side'
    )
