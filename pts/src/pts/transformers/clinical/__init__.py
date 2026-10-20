"""Clinical report datasets, providers and the AACT LLM extraction.

Production code outside this package imports from here, not from its submodules.
"""

from pts.transformers.clinical.datasets import (
    CATEGORY_RANKS,
    CATEGORY_RANKS_STR,
    RANK_TO_CATEGORY_STR,
    ClinicalIndication,
    ClinicalReport,
    ClinicalReportExtraction,
)
from pts.transformers.clinical.helpers import filter_df, union_dfs
from pts.transformers.clinical.mapping import map_entities
from pts.transformers.clinical.ner import extract_disease_entities, extract_drug_entities
from pts.transformers.clinical.providers.aact import extract_clinical_report as extract_aact_clinical_report
from pts.transformers.clinical.providers.aact import replace_with_llm_indications
from pts.transformers.clinical.providers.aact_llm import build_prompts, sample_report
from pts.transformers.clinical.providers.chembl_drug_warnings import (
    extract_clinical_report as extract_drug_warning_clinical_report,
)
from pts.transformers.clinical.providers.chembl_indications import (
    extract_clinical_report as extract_chembl_clinical_report,
)
from pts.transformers.clinical.providers.ema import extract_clinical_report as extract_ema_clinical_report
from pts.transformers.clinical.providers.pmda import extract_clinical_report as extract_pmda_clinical_report
from pts.transformers.clinical.providers.pmda import parse_pmda_approvals
from pts.transformers.clinical.providers.ttd import extract_clinical_report as extract_ttd_clinical_report
from pts.transformers.clinical.providers.ttd import extract_indication as extract_ttd_indication
from pts.transformers.clinical.schemas import (
    ClinicalProvider,
    ClinicalReportExtractionSchema,
    ClinicalReportOrigin,
    ClinicalReportType,
    ClinicalSource,
    ClinicalStageCategory,
)

__all__ = [
    'CATEGORY_RANKS',
    'CATEGORY_RANKS_STR',
    'RANK_TO_CATEGORY_STR',
    'ClinicalIndication',
    'ClinicalProvider',
    'ClinicalReport',
    'ClinicalReportExtraction',
    'ClinicalReportExtractionSchema',
    'ClinicalReportOrigin',
    'ClinicalReportType',
    'ClinicalSource',
    'ClinicalStageCategory',
    'build_prompts',
    'extract_aact_clinical_report',
    'extract_chembl_clinical_report',
    'extract_disease_entities',
    'extract_drug_entities',
    'extract_drug_warning_clinical_report',
    'extract_ema_clinical_report',
    'extract_pmda_clinical_report',
    'extract_ttd_clinical_report',
    'extract_ttd_indication',
    'filter_df',
    'map_entities',
    'parse_pmda_approvals',
    'replace_with_llm_indications',
    'sample_report',
    'union_dfs',
]
