"""The vocabularies of the credible set stages match gentropy and the PRDs."""

from pts.schemas.credible_set import Chromosome, FinemappingMethod
from pts.schemas.study import StudyType
from pts.transformers.credible_set.region import KnownGenomicRegion
from pts.transformers.utils.ensembl.queries import CANONICAL_CHROMOSOMES


def test_finemapping_methods_match_gentropy() -> None:
    assert [method.value for method in FinemappingMethod] == ['PICS', 'SuSiE', 'SuSiE-inf']
    assert FinemappingMethod.susie() == (FinemappingMethod.SUSIE, FinemappingMethod.SUSIE_INF)


def test_chromosomes_are_the_ensembl_canonical_chromosomes_in_genome_order() -> None:
    """The genes and variants credible sets are joined to lie on these, so there is no XY."""
    assert [chromosome.value for chromosome in Chromosome] == CANONICAL_CHROMOSOMES


def test_mhc_matches_gentropy() -> None:
    mhc = KnownGenomicRegion.MHC.value
    assert (mhc.chromosome, mhc.start, mhc.end) == (Chromosome.CHR_6, 25726063, 33400556)


def test_study_types_match_gentropy() -> None:
    assert [study_type.value for study_type in StudyType] == [
        'gwas',
        'eqtl',
        'pqtl',
        'sqtl',
        'tuqtl',
        'sceqtl',
        'scpqtl',
        'scsqtl',
        'sctuqtl',
    ]
