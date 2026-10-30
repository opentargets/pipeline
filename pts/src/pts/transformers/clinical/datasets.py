"""Clinical report, clinical indication and LLM extraction datasets."""

from collections.abc import Callable

import polars as pl
import polars_hash as plh

from pts.transformers.clinical.helpers import sanitise_text
from pts.transformers.clinical.schemas import (
    ClinicalIndicationSchema,
    ClinicalProvider,
    ClinicalReportExtractionSchema,
    ClinicalReportSchema,
    ClinicalStageCategory,
    snake_to_camel,
    validate_schema,
)

# Category ranking for Maximum Clinical Development Status
CATEGORY_RANKS = {
    ClinicalStageCategory.WITHDRAWAL: 1,
    ClinicalStageCategory.PHASE_4: 2,
    ClinicalStageCategory.APPROVAL: 3,
    ClinicalStageCategory.PREAPPROVAL: 4,
    ClinicalStageCategory.PHASE_3: 5,
    ClinicalStageCategory.PHASE_2_3: 6,
    ClinicalStageCategory.PHASE_2: 7,
    ClinicalStageCategory.PHASE_1_2: 8,
    ClinicalStageCategory.PHASE_1: 9,
    ClinicalStageCategory.EARLY_PHASE_1: 10,
    ClinicalStageCategory.IND: 11,
    ClinicalStageCategory.PRECLINICAL: 12,
    ClinicalStageCategory.UNKNOWN: 13,
}

CATEGORY_RANKS_STR = {k.value: v for k, v in CATEGORY_RANKS.items()}
RANK_TO_CATEGORY_STR = {v: k.value for k, v in CATEGORY_RANKS.items()}

# Clinical status harmonization constants
PHASE_TO_CATEGORY_MAP = {
    # WITHDRAWAL (Rank 1)
    'withdrawn': ClinicalStageCategory.WITHDRAWAL,
    'withdrawn from market': ClinicalStageCategory.WITHDRAWAL,
    'revoked': ClinicalStageCategory.WITHDRAWAL,
    'expired': ClinicalStageCategory.WITHDRAWAL,
    'lapsed': ClinicalStageCategory.WITHDRAWAL,
    'suspended': ClinicalStageCategory.WITHDRAWAL,
    # PHASE_4 (Rank 2)
    'phase4': ClinicalStageCategory.PHASE_4,
    'phase 4': ClinicalStageCategory.PHASE_4,
    'discontinued in phase 4': ClinicalStageCategory.PHASE_4,
    '4.0': ClinicalStageCategory.PHASE_4,
    # APPROVAL (Rank 3)
    'approved': ClinicalStageCategory.APPROVAL,
    'authorised': ClinicalStageCategory.APPROVAL,
    'approved (orphan drug)': ClinicalStageCategory.APPROVAL,
    'approved in china': ClinicalStageCategory.APPROVAL,
    'approved in eu': ClinicalStageCategory.APPROVAL,
    'registered': ClinicalStageCategory.APPROVAL,
    # PREAPPROVAL (Rank 4)
    'preregistration': ClinicalStageCategory.PREAPPROVAL,
    'application submitted': ClinicalStageCategory.PREAPPROVAL,
    'approval submitted': ClinicalStageCategory.PREAPPROVAL,
    'nda filed': ClinicalStageCategory.PREAPPROVAL,
    'bla submitted': ClinicalStageCategory.PREAPPROVAL,
    'opinion': ClinicalStageCategory.PREAPPROVAL,
    'opinion under re-examination': ClinicalStageCategory.PREAPPROVAL,
    'discontinued in preregistration': ClinicalStageCategory.PREAPPROVAL,
    # PHASE_3 (Rank 5)
    'phase3': ClinicalStageCategory.PHASE_3,
    'phase 3': ClinicalStageCategory.PHASE_3,
    'discontinued in phase 3': ClinicalStageCategory.PHASE_3,
    '3.0': ClinicalStageCategory.PHASE_3,
    # PHASE_2_3 (Rank 6)
    'phase2/phase3': ClinicalStageCategory.PHASE_2_3,
    'phase 2/3': ClinicalStageCategory.PHASE_2_3,
    'discontinued in phase 2/3': ClinicalStageCategory.PHASE_2_3,
    # PHASE_2 (Rank 7)
    'phase2': ClinicalStageCategory.PHASE_2,
    'phase 2': ClinicalStageCategory.PHASE_2,
    'phase 2a': ClinicalStageCategory.PHASE_2,
    'phase 2b': ClinicalStageCategory.PHASE_2,
    'discontinued in phase 2': ClinicalStageCategory.PHASE_2,
    'discontinued in phase 2a': ClinicalStageCategory.PHASE_2,
    'discontinued in phase 2b': ClinicalStageCategory.PHASE_2,
    '2.0': ClinicalStageCategory.PHASE_2,
    # PHASE_1_2 (Rank 8)
    'phase1/phase2': ClinicalStageCategory.PHASE_1_2,
    'phase 1/2': ClinicalStageCategory.PHASE_1_2,
    'phase 1b/2a': ClinicalStageCategory.PHASE_1_2,
    'phase 1/2a': ClinicalStageCategory.PHASE_1_2,
    'discontinued in phase 1/2': ClinicalStageCategory.PHASE_1_2,
    # PHASE_1 (Rank 9)
    'phase1': ClinicalStageCategory.PHASE_1,
    'phase 1': ClinicalStageCategory.PHASE_1,
    'phase 1b': ClinicalStageCategory.PHASE_1,
    'discontinued in phase 1': ClinicalStageCategory.PHASE_1,
    '1.0': ClinicalStageCategory.PHASE_1,
    # EARLY_PHASE_1 (Rank 10)
    'early_phase1': ClinicalStageCategory.EARLY_PHASE_1,
    'phase 0': ClinicalStageCategory.EARLY_PHASE_1,
    '0.5': ClinicalStageCategory.EARLY_PHASE_1,
    # IND (Rank 11)
    'ind submitted': ClinicalStageCategory.IND,
    'investigative': ClinicalStageCategory.IND,
    # PRECLINICAL (Rank 12)
    'preclinical': ClinicalStageCategory.PRECLINICAL,
    'patented': ClinicalStageCategory.PRECLINICAL,
    # UNKNOWN (Rank 13)
    'clinical trial': ClinicalStageCategory.UNKNOWN,
    'terminated': ClinicalStageCategory.UNKNOWN,
    'application withdrawn': ClinicalStageCategory.UNKNOWN,
    'refused': ClinicalStageCategory.UNKNOWN,
    'withdrawn from rolling review': ClinicalStageCategory.UNKNOWN,
    'NA': ClinicalStageCategory.UNKNOWN,
    '-1.0': ClinicalStageCategory.UNKNOWN,
}


# Sources that indicate approved status when phase is null
APPROVAL_SOURCES = {'ATC', 'EMA', 'FDA', 'DailyMed', 'PMDA'}


def map_phase_to_category(phase: str | None, source: str, overall_status: str | None) -> ClinicalStageCategory:
    """Map original phase value to standardised category.

    Args:
        phase: Original phase value (can be null)
        source: Data source name
        overall_status: Overall status of the trial. Relevant when it reports approvals for compassionate use cases.

    Returns:
        Standardised clinical status category
    """
    withdrawn_values = [k for k, v in PHASE_TO_CATEGORY_MAP.items() if v == ClinicalStageCategory.WITHDRAWAL]
    if phase in withdrawn_values:
        return ClinicalStageCategory.WITHDRAWAL
    elif source in APPROVAL_SOURCES:
        return ClinicalStageCategory.APPROVAL
    elif overall_status == 'APPROVED_FOR_MARKETING':
        return ClinicalStageCategory.APPROVAL

    # Handle case-insensitive mapping
    phase_lower = phase.lower() if isinstance(phase, str) else str(phase).lower()

    return PHASE_TO_CATEGORY_MAP.get(phase_lower, ClinicalStageCategory.UNKNOWN)


class ClinicalReport:
    """A dataset for clinical reports (e.g. clinical trial, an USAN reference), wrapping a Polars DataFrame."""

    def __init__(self, df: pl.DataFrame):
        """Initialises the dataset, validating and aligning the DataFrame."""
        # Harmonise column names from snake to camel case
        df = df.rename({col: snake_to_camel(col) for col in df.columns})

        # Create struct with optional trialOverallStatus column
        struct_expr = pl.struct(['phaseFromSource', 'source'])
        if 'trialOverallStatus' in df.columns:
            struct_expr = pl.struct(['phaseFromSource', 'source', 'trialOverallStatus'])

        df = df.with_columns(
            # Assign clinical stage
            clinicalStage=struct_expr.map_elements(
                lambda row: map_phase_to_category(
                    row['phaseFromSource'],
                    row['source'],
                    row.get('trialOverallStatus'),  # Safe get for optional column
                ),
                return_dtype=pl.String,
            ),
            id=pl.col('id').str.to_lowercase(),
        )

        # Drop duplicates by id, keeping the row with the best clinical stage
        df = self.drop_duplicates(df)

        df = df.with_columns(
            # Apply sanitise_text to all top-level string columns. Strings nested
            # inside structs or lists (drugs, diseases, sideEffects, countries) are
            # not reached by `pl.col(pl.String)`.
            pl.col(pl.String).map_elements(sanitise_text, return_dtype=pl.String)
        )
        self.df = validate_schema(df, ClinicalReportSchema)

    def pipe(self, func: Callable, *args, **kwargs) -> 'ClinicalReport':
        """Apply a function to the dataset."""
        return func(self, *args, **kwargs)

    @classmethod
    def drop_duplicates(cls, df: pl.DataFrame) -> pl.DataFrame:
        """Drop duplicate reports based on clinical stage.

        When multiple rows share the same ``id`` and have equal clinical
        stage, prefer the report whose provider is the first-party
        distributor of its source (``ClinicalProvider.owns_source``).
        This is relevant for EMA references, where both `EMA` and `ChEMBL` providers
        capture the same IDs.
        """
        return (
            df
            .with_columns(
                clinicalStageRank=pl.col('clinicalStage').replace_strict(
                    CATEGORY_RANKS_STR,
                    default=CATEGORY_RANKS_STR[ClinicalStageCategory.UNKNOWN.value],
                ),
                providerPriority=pl.struct(['provider', 'source']).map_elements(
                    lambda r: (
                        0
                        if r['provider'] is not None and ClinicalProvider.owns_source(r['provider'], r['source'])
                        else 1
                    ),
                    return_dtype=pl.Int8,
                ),
            )
            .sort(['id', 'clinicalStageRank', 'providerPriority'])
            .unique(subset=['id'], keep='first')
            .drop('clinicalStageRank', 'providerPriority')
        )


class ClinicalIndication:
    """A dataset for drug-indication relationships, wrapping a Polars DataFrame."""

    AGGREGATION_FIELDS = {
        'id',
        'drugName',
        'diseaseName',
        'drugId',
        'diseaseId',
    }

    def __init__(self, df: pl.DataFrame):
        """Initialises the dataset, validating and aligning the DataFrame.

        Also assigns mapping status and maximum clinical status.
        """
        df = df.with_columns(
            mappingStatus=self._assign_mapping_status(pl.col('drugId'), pl.col('diseaseId')),
        )
        self.df = validate_schema(df, ClinicalIndicationSchema)

    @classmethod
    def _get_study_metadata_columns(cls, df: pl.DataFrame) -> list[str]:
        """Get all columns that represent study metadata (everything except aggregation fields)."""
        return sorted(set(df.columns) - cls.AGGREGATION_FIELDS)

    @classmethod
    def from_report(cls, report: pl.DataFrame) -> 'ClinicalIndication':
        """Aggregate drug/disease information collected in clinical report into a ClinicalIndication."""
        # Assert validity of reports
        report = validate_schema(report, ClinicalReportSchema)

        return cls(
            df=(
                # Explode clinical reports to get lists of study/disease/drug
                report
                .explode('drugs', empty_as_null=True)
                .explode('diseases', empty_as_null=True)
                .unnest(['drugs', 'diseases'])
                .rename({'id': 'clinicalReportId'})
                .with_columns(
                    clinicalStageForMax=pl
                    .when(
                        pl.col('clinicalStage').is_in([
                            ClinicalStageCategory.WITHDRAWAL.value,
                            ClinicalStageCategory.PHASE_4.value,
                            ClinicalStageCategory.APPROVAL.value,
                        ])
                    )
                    .then(pl.lit(ClinicalStageCategory.APPROVAL.value))
                    .otherwise(pl.col('clinicalStage')),
                )
                .with_columns(
                    # Create hash to use as primary key (no studyId in this case)
                    id=plh.concat_str(
                        pl.coalesce(pl.col('drugId'), pl.col('drugFromSource')),
                        pl.coalesce(pl.col('diseaseId'), pl.col('diseaseFromSource')),
                    ).chash.sha2_256(),
                    drugName=pl.coalesce(pl.col('drugId'), pl.col('drugFromSource')),
                    diseaseName=pl.coalesce(pl.col('diseaseId'), pl.col('diseaseFromSource')),
                    # Map clinicalStage to rank for sorting
                    clinicalStageRank=pl.col('clinicalStageForMax').replace_strict(
                        CATEGORY_RANKS_STR,
                        default=CATEGORY_RANKS[ClinicalStageCategory.UNKNOWN],
                    ),
                )
                .sort('clinicalStageRank')
                .group_by(list(cls.AGGREGATION_FIELDS), maintain_order=True)
                .agg(
                    pl.col('clinicalReportId').unique().alias('clinicalReportIds'),
                    # Get the maximum clinical stage (minimum rank = highest priority)
                    pl.col('clinicalStageForMax').first().alias('maxClinicalStage'),
                )
            ),
        )

    @staticmethod
    def _assign_mapping_status(drug_id_column: pl.Expr, disease_id_column: pl.Expr) -> pl.Expr:
        return (
            pl
            .when((drug_id_column.is_not_null()) & (disease_id_column.is_not_null()))
            .then(pl.lit('FULLY_MAPPED'))
            .when((drug_id_column.is_not_null()) & (disease_id_column.is_null()))
            .then(pl.lit('DRUG_MAPPED'))
            .when((drug_id_column.is_null()) & (disease_id_column.is_not_null()))
            .then(pl.lit('DISEASE_MAPPED'))
            .otherwise(pl.lit('UNMAPPED'))
        )

    def filter_by_report_id(self, report_id: str) -> 'ClinicalIndication':
        """Get associations supported by a given report ID."""
        return ClinicalIndication(df=self.df.filter(pl.col('clinicalReportIds').list.contains(report_id)))


class ClinicalReportExtraction:
    """A validated collection of LLM-extracted clinical trial data.

    The wrapped DataFrame conforms to the ``ClinicalReportExtractionSchema``.
    """

    def __init__(self, df: pl.DataFrame):
        if not df.is_empty():
            df = validate_schema(df, ClinicalReportExtractionSchema)
        self.df = df
