"""Response schemas for the mechanism extraction and its judge, and their prompts."""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from importlib import resources
from typing import Literal

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

PROMPTS_PACKAGE = 'pts.transformers.clinical_mira.moa.prompts'


class ActionType(StrEnum):
    """ChEMBL action types, as in the Platform ``drug_mechanism_of_action`` dataset."""

    ACTIVATOR = 'ACTIVATOR'
    AGONIST = 'AGONIST'
    ALLOSTERIC_ANTAGONIST = 'ALLOSTERIC ANTAGONIST'
    ANTAGONIST = 'ANTAGONIST'
    ANTISENSE_INHIBITOR = 'ANTISENSE INHIBITOR'
    BINDING_AGENT = 'BINDING AGENT'
    BLOCKER = 'BLOCKER'
    CROSS_LINKING_AGENT = 'CROSS-LINKING AGENT'
    DEGRADER = 'DEGRADER'
    DISRUPTING_AGENT = 'DISRUPTING AGENT'
    EXOGENOUS_GENE = 'EXOGENOUS GENE'
    EXOGENOUS_PROTEIN = 'EXOGENOUS PROTEIN'
    GENE_EDITING_NEGATIVE_MODULATOR = 'GENE EDITING NEGATIVE MODULATOR'
    HYDROLYTIC_ENZYME = 'HYDROLYTIC ENZYME'
    INHIBITOR = 'INHIBITOR'
    INVERSE_AGONIST = 'INVERSE AGONIST'
    MODULATOR = 'MODULATOR'
    NEGATIVE_ALLOSTERIC_MODULATOR = 'NEGATIVE ALLOSTERIC MODULATOR'
    NEGATIVE_MODULATOR = 'NEGATIVE MODULATOR'
    OPENER = 'OPENER'
    PARTIAL_AGONIST = 'PARTIAL AGONIST'
    POSITIVE_ALLOSTERIC_MODULATOR = 'POSITIVE ALLOSTERIC MODULATOR'
    POSITIVE_MODULATOR = 'POSITIVE MODULATOR'
    PROTEOLYTIC_ENZYME = 'PROTEOLYTIC ENZYME'
    RELEASING_AGENT = 'RELEASING AGENT'
    RNAI_INHIBITOR = 'RNAI INHIBITOR'
    STABILISER = 'STABILISER'
    SUBSTRATE = 'SUBSTRATE'
    VACCINE_ANTIGEN = 'VACCINE ANTIGEN'
    OTHER = 'OTHER'


class TargetType(StrEnum):
    """ChEMBL target types, as in the Platform ``drug_mechanism_of_action`` dataset."""

    CHIMERIC_PROTEIN = 'chimeric protein'
    NUCLEIC_ACID = 'nucleic-acid'
    PROTEIN_COMPLEX = 'protein complex'
    PROTEIN_COMPLEX_GROUP = 'protein complex group'
    PROTEIN_FAMILY = 'protein family'
    PROTEIN_NUCLEIC_ACID_COMPLEX = 'protein nucleic-acid complex'
    PROTEIN_PROTEIN_INTERACTION = 'protein-protein interaction'
    SELECTIVITY_GROUP = 'selectivity group'
    SINGLE_PROTEIN = 'single protein'
    OTHER = 'OTHER'


class _Strict(BaseModel):
    model_config = ConfigDict(extra='forbid', alias_generator=to_camel, validate_by_name=True, validate_by_alias=True)


class Evidence(_Strict):
    """A verbatim quote and the trial it was copied from."""

    nct_id: str
    quote: str


class Mechanism(_Strict):
    """One mechanism of action, as stated in the trial texts."""

    target_from_source: str
    target_symbol_suggestion: list[str]
    is_human_target: bool
    acts_on_receptor: bool
    relation: Literal['direct', 'changes_amount', 'cells_or_marker', 'transport_or_resistance']
    statement_type: Literal['drug_description', 'experimental_finding', 'hypothesis']
    target_type: TargetType
    action_type: ActionType
    mechanism_of_action: str
    evidence: list[Evidence]
    confidence: float


class MoaExtraction(_Strict):
    """The mechanisms the pooled trial texts state for one drug."""

    drug_name: str
    not_stated: bool
    mechanisms: list[Mechanism]


class JudgeVerdict(_Strict):
    """A pharmacologist's verdict on one extracted mechanism."""

    target_supported: Literal['yes', 'no', 'unclear']
    action_supported: Literal['yes', 'no', 'unclear']
    primary_mechanism: Literal['yes', 'no', 'unclear']
    rationale: str


RESPONSE_MODELS: dict[str, type[BaseModel]] = {
    'extract': MoaExtraction,
    'extract_papers': MoaExtraction,
    'judge': JudgeVerdict,
}


def load_prompt(name: Literal['extract', 'extract_papers', 'judge']) -> str:
    """Return the packaged system prompt of the extractor or the judge."""
    return (resources.files(PROMPTS_PACKAGE) / f'{name}.txt').read_text(encoding='utf-8')


def papers_prompt() -> str:
    """Return the extractor's system prompt followed by the publications addendum."""
    return load_prompt('extract') + load_prompt('extract_papers')


def ruleset_sha256(name: Literal['extract', 'extract_papers', 'judge']) -> str:
    """Return the sha256 of a stage's system prompt and response schema."""
    schema = json.dumps(RESPONSE_MODELS[name].model_json_schema(by_alias=True), sort_keys=True)
    return hashlib.sha256(f'{load_prompt(name)}\x1f{schema}'.encode()).hexdigest()
