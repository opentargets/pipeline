"""Guards on the evidence schema and its Polars->Spark conversion.

`pts/schemas/evidence.py` replaced a spark `evidence.json` that both the polars transformers and
the pyspark `association` job read. Two properties have to survive that: the polars dtypes the
transformers pin, and the `StructType` spark parses with. Neither engine errors when the schema
drifts -- polars silently changes which duplicate survives, spark silently changes what it
parses -- so both are pinned here as data.
"""

from __future__ import annotations

import polars as pl
import pytest
from pyspark.sql.types import (
    ArrayType,
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from pts.pyspark.common.utils import polars_schema_to_spark, polars_type_to_spark
from pts.schemas.evidence import evidence_schema

# The full field order, pinned deliberately rather than spot-checked. This dict's key order
# becomes the frame's column order in `_harmonise_to_schema`, and column order feeds the content
# hash that picks the surviving row in `Evidence.validate_uniqueness` -- so a reordering is a
# silent output change, and on this port that failure mode has already occurred three times.
EXPECTED_FIELD_ORDER = (
    'id', 'targetFromSourceId', 'diseaseFromSourceMappedId', 'actionType', 'biomarkerName', 'biomarkers',
    'confidence', 'datasourceId', 'datatypeId', 'diseaseFromSource', 'drugFromSource', 'drugId', 'drugResponse',
    'literature', 'urls', 'qualityControls', 'diseaseId', 'targetId', 'curationDate', 'publicationDate',
    'evidenceDate', 'score', 'diseaseFromSourceId', 'mutatedSamples', 'resourceScore', 'studyId',
    'directionOnTrait', 'directionOnTarget', 'clinicalReportId', 'clinicalStage', 'cohortPhenotypes',
    'studyStartDate', 'trialWhyStopped', 'trialStopReasonCategories', 'targetFromSource', 'allelicRequirements',
    'releaseDate', 'diseaseCellLines', 'cellType', 'contrast', 'crisprScreenLibrary', 'geneticBackground',
    'log2FoldChangeValue', 'projectId', 'statisticalTestTail', 'studyOverview', 'biomarkerList',
    'geneInteractionType', 'geneticInteractionPValue', 'geneticInteractionScore',
    'interactingTargetFromSourceId', 'interactingTargetRole', 'phenotypicConsequenceFDR',
    'phenotypicConsequenceLogFoldChange', 'phenotypicConsequencePValue', 'projectDescription', 'releaseVersion',
    'statisticalMethod', 'targetRole', 'pmcIds', 'publicationYear', 'textMiningSentences', 'alleleOrigins',
    'clinicalSignificances', 'variantFromSourceId', 'variantFunctionalConsequenceId', 'variantHgvsId',
    'variantId', 'variantRsId', 'biosamplesFromSource', 'log2FoldChangePercentileRank', 'ancestry',
    'ancestryId', 'beta', 'betaConfidenceIntervalLower', 'betaConfidenceIntervalUpper', 'cohortId', 'oddsRatio',
    'oddsRatioConfidenceIntervalLower', 'oddsRatioConfidenceIntervalUpper', 'pValueExponent', 'pValueMantissa',
    'sex', 'statisticalMethodOverview', 'studyCases', 'studyCasesWithQualifyingVariants', 'studySampleSize',
    'studyLocusId', 'biologicalModelAllelicComposition', 'biologicalModelGeneticBackground',
    'biologicalModelId', 'diseaseModelAssociatedHumanPhenotypes', 'diseaseModelAssociatedModelPhenotypes',
    'targetInModel', 'targetInModelEnsemblId', 'targetInModelMgiId', 'cohortDescription', 'cohortShortName',
    'significantDriverMethods', 'cellLineBackground', 'assays', 'assessments', 'primaryProjectHit',
    'primaryProjectId', 'pathways', 'reactionId', 'reactionName', 'targetModulation',
    'variantAminoacidDescriptions',
)


class TestEvidenceSchema:
    def test_field_order_is_pinned(self) -> None:
        """Not just the field SET -- the order, which decides the de-duplication survivor."""
        assert tuple(evidence_schema) == EXPECTED_FIELD_ORDER

    def test_field_count(self) -> None:
        assert len(evidence_schema) == 109

    def test_representative_scalar_dtypes(self) -> None:
        assert evidence_schema['datasourceId'] == pl.String
        assert evidence_schema['score'] == pl.Float64
        # long, not Int32: `_harmonise_to_schema` casts to this, and a narrower type would
        # overflow rather than error on real gwas rows.
        assert evidence_schema['pValueExponent'] == pl.Int64

    def test_list_and_two_level_nesting_survive(self) -> None:
        """`biomarkers` is the deepest shape in the schema: a Struct of List(Struct)."""
        assert evidence_schema['literature'] == pl.List(pl.String)

        biomarkers = evidence_schema['biomarkers']
        assert isinstance(biomarkers, pl.Struct)
        gene_expression = dict(biomarkers.to_schema())['geneExpression']
        assert gene_expression == pl.List(pl.Struct({'id': pl.String, 'name': pl.String}))


class TestPolarsTypeToSpark:
    @pytest.mark.parametrize(
        ('polars_type', 'expected'),
        [
            (pl.String, StringType()),
            (pl.Float64, DoubleType()),
            (pl.Int64, LongType()),
            (pl.List(pl.String), ArrayType(StringType())),
        ],
    )
    def test_scalar_and_list(self, polars_type: object, expected: object) -> None:
        assert polars_type_to_spark(polars_type) == expected

    def test_struct(self) -> None:
        converted = polars_type_to_spark(pl.Struct({'id': pl.String, 'n': pl.Int64}))

        assert converted == StructType([
            StructField('id', StringType(), nullable=True),
            StructField('n', LongType(), nullable=True),
        ])

    def test_struct_of_list_of_struct(self) -> None:
        """The `biomarkers` shape -- the case a one-level converter would silently mangle."""
        converted = polars_type_to_spark(pl.Struct({'g': pl.List(pl.Struct({'id': pl.String}))}))

        assert converted == StructType([
            StructField(
                'g',
                ArrayType(StructType([StructField('id', StringType(), nullable=True)])),
                nullable=True,
            ),
        ])

    def test_unsupported_type_raises(self) -> None:
        """Rather than substituting a plausible type -- a wrong read schema changes what spark parses."""
        with pytest.raises(ValueError, match='Unsupported Polars type'):
            polars_type_to_spark(pl.Categorical)


class TestPolarsSchemaToSpark:
    def test_evidence_schema_converts_wholesale(self) -> None:
        converted = polars_schema_to_spark(evidence_schema)

        assert len(converted.fields) == 109
        assert tuple(field.name for field in converted) == EXPECTED_FIELD_ORDER

    def test_every_field_is_nullable(self) -> None:
        """Spark's `StructType.fromJson` defaulted every evidence field to nullable.

        Narrowing any of them would make spark reject rows it reads today.
        """
        assert all(field.nullable for field in polars_schema_to_spark(evidence_schema))

    def test_field_order_is_preserved(self) -> None:
        converted = polars_schema_to_spark({'z': pl.String, 'a': pl.Int64})

        assert [field.name for field in converted] == ['z', 'a']
