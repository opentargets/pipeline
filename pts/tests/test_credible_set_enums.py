"""The vocabularies of the credible set stages match gentropy and the PRDs."""

from pts.schemas.study import StudyType
from pts.transformers.credible_set.enums import Chromosome, FinemappingMethod, KnownGenomicRegion


def test_finemapping_methods_match_gentropy() -> None:
    assert [method.value for method in FinemappingMethod] == ['PICS', 'SuSiE', 'SuSiE-inf']
    assert FinemappingMethod.susie() == (FinemappingMethod.SUSIE, FinemappingMethod.SUSIE_INF)


def test_chromosomes_are_listed_in_genome_order() -> None:
    assert [chromosome.value for chromosome in Chromosome] == [
        *(str(number) for number in range(1, 23)),
        'X',
        'Y',
        'XY',
        'MT',
    ]


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
