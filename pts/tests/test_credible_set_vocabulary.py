"""The vocabularies of the credible set stages match gentropy and the PRDs."""

from typing import get_args

from ot_datamodels.vocabularies import StudyType

from pts.schemas.credible_set import CHROMOSOMES, FINEMAPPING_METHODS, SUSIE_METHODS
from pts.transformers.credible_set.region import KnownGenomicRegion
from pts.transformers.utils.ensembl.queries import CANONICAL_CHROMOSOMES


def test_finemapping_methods_match_gentropy() -> None:
    assert FINEMAPPING_METHODS == ('PICS', 'SuSiE', 'SuSiE-inf')
    assert SUSIE_METHODS == ('SuSiE', 'SuSiE-inf')


def test_chromosomes_are_the_ensembl_canonical_chromosomes_in_genome_order() -> None:
    """The genes and variants credible sets are joined to lie on these, so there is no XY."""
    assert list(CHROMOSOMES) == CANONICAL_CHROMOSOMES


def test_mhc_matches_gentropy() -> None:
    mhc = KnownGenomicRegion.MHC.value
    assert (mhc.chromosome, mhc.start, mhc.end) == ('6', 25726063, 33400556)


def test_study_types_match_gentropy() -> None:
    """gentropy's `StudyType`; the order is ClickHouse's Enum's, not gentropy's."""
    study_types = {'gwas', 'eqtl', 'pqtl', 'sqtl', 'tuqtl', 'sceqtl', 'scpqtl', 'scsqtl', 'sctuqtl'}
    assert set(get_args(StudyType)) == study_types
