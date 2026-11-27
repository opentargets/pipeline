"""The QUALIFIED flags of credible sets, from the lead variant and the study's qualification."""

from typing import Any

import polars as pl
import pytest

from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.qualification import flag_qualified_sets
from pts.transformers.study import QualityFlag as StudyQualityFlag

DISEASE = StudyQualityFlag.QUALIFIED_DISEASE_STUDY.value
MEASUREMENT = StudyQualityFlag.QUALIFIED_MEASUREMENT_STUDY.value


def _flags(lead: dict[str, Any], study: dict[str, Any]) -> list[str]:
    """The flags of one set, whose lead row and study default to a qualified disease set."""
    credible_set = pl.LazyFrame({
        'id': ['cs'],
        'studyId': ['s'],
        'leadVariantId': ['v'],
        'qualityControls': [['existing']],
    })
    lead = {'credibleSetId': 'cs', 'variantId': 'v', 'effectAlleleFrequency': 0.2, 'rescaledAbsBeta': 0.1, **lead}
    variant = pl.LazyFrame(
        [lead],
        schema={
            'credibleSetId': pl.String,
            'variantId': pl.String,
            'effectAlleleFrequency': pl.Float32,
            'rescaledAbsBeta': pl.Float32,
        },
        orient='row',
    )
    study = {'studyId': 's', 'nCases': 1_000, 'nSamples': 10_000, 'qualityControls': [DISEASE], **study}
    study_frame = pl.LazyFrame(
        [study],
        schema={'studyId': pl.String, 'nCases': pl.Int32, 'nSamples': pl.Int32, 'qualityControls': pl.List(pl.String)},
        orient='row',
    )
    return flag_qualified_sets(credible_set, variant, study_frame).collect()['qualityControls'].to_list()[0]


@pytest.mark.parametrize(
    ('lead', 'study', 'flag'),
    [
        ({}, {}, CredibleSetFlag.QUALIFIED_DISEASE_SET),
        ({}, {'qualityControls': []}, None),
        ({'effectAlleleFrequency': 0.0101}, {}, CredibleSetFlag.QUALIFIED_DISEASE_SET),
        ({'effectAlleleFrequency': 0.0099}, {}, None),
        ({'effectAlleleFrequency': 0.9899}, {}, CredibleSetFlag.QUALIFIED_DISEASE_SET),
        ({'effectAlleleFrequency': 0.995}, {}, None),
        ({'effectAlleleFrequency': None}, {}, None),
        ({'rescaledAbsBeta': 3.0}, {}, CredibleSetFlag.QUALIFIED_DISEASE_SET),
        ({'rescaledAbsBeta': 3.01}, {}, None),
        ({'rescaledAbsBeta': None}, {}, None),
        ({'variantId': 'other'}, {}, None),
        (
            {'effectAlleleFrequency': 0.00101},
            {'qualityControls': [MEASUREMENT]},
            CredibleSetFlag.QUALIFIED_MEASUREMENT_SET,
        ),
        ({'effectAlleleFrequency': 0.0009}, {'qualityControls': [MEASUREMENT]}, None),
        ({}, {'qualityControls': [MEASUREMENT], 'nCases': None}, CredibleSetFlag.QUALIFIED_MEASUREMENT_SET),
    ],
    ids=[
        'disease',
        'study-not-qualified',
        'disease-mac-above-20',
        'disease-mac-below-20',
        'minor-allele-is-other-mac-above-20',
        'minor-allele-is-other-mac-below-20',
        'no-frequency',
        'beta-3',
        'beta-above-3',
        'no-beta',
        'no-lead-row',
        'measurement-mac-above-20-from-samples',
        'measurement-mac-below-20',
        'measurement-without-cases',
    ],
)
def test_flag_qualified_sets(lead: dict[str, Any], study: dict[str, Any], flag: CredibleSetFlag | None) -> None:
    """A set of a qualified study qualifies when its lead has the minor allele count and effect size."""
    assert _flags(lead, study) == sorted(['existing', *([flag] if flag else [])])
