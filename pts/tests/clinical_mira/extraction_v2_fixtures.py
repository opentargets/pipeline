"""Frames of v2 trial extractions, shaped like the published snapshot."""

import polars as pl

from pts.transformers.clinical_mira.model_frames import polars_schema
from pts.transformers.clinical_mira.schemas import TrialExtractionV2

V2_SCHEMA = pl.Schema({'id': pl.String, 'prompt_sha256': pl.String, **polars_schema(TrialExtractionV2)})


def v2_link(drug, indication, confidence=0.9, broader=None, synonyms=()):
    return {
        'drug': drug,
        'synonyms': list(synonyms),
        'indication': indication,
        'broader': broader or indication,
        'use': 'treat',
        'arms': [],
        'confidence': confidence,
        'evidence_quote': 'q',
    }


def v2_role(drug, kind, synonyms=()):
    return {'drug': drug, 'synonyms': list(synonyms), 'role': kind, 'evidence_quote': 'q'}


def v2_frame(rows: dict[str, dict]) -> pl.DataFrame:
    full = [
        {
            'id': trial_id,
            'prompt_sha256': 'x',
            'drug_intent': 'therapeutic',
            'drug_intent_confidence': 0.9,
            'links': [],
            'background_drugs': [],
            'other_drugs': [],
            'population_conditions': [],
            **row,
        }
        for trial_id, row in rows.items()
    ]
    return pl.DataFrame(full, schema=V2_SCHEMA)
