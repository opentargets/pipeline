"""The vocabularies of the credible set stages match gentropy and the PRDs."""

from pts.schemas.credible_set import CHROMOSOMES, FINEMAPPING_METHODS, SUSIE_METHODS
from pts.schemas.study import STUDY_TYPES
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
    assert set(STUDY_TYPES) == {'gwas', 'eqtl', 'pqtl', 'sqtl', 'tuqtl', 'sceqtl', 'scpqtl', 'scsqtl', 'sctuqtl'}
