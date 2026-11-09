"""The fixed vocabularies of credible sets: fine-mapping methods, chromosomes and known regions.

The validated and published credible sets type `fineMappingMethod` and `chromosome` as enums of
these, and set QC flags a set outside them. The known regions are those a rule treats apart, such
as the MHC, whose long-range LD makes its credible sets unreliable.
"""

from __future__ import annotations

from enum import Enum, StrEnum
from typing import NamedTuple


class FinemappingMethod(StrEnum):
    """Fine-mapping methods."""

    PICS = 'PICS'
    SUSIE = 'SuSiE'
    SUSIE_INF = 'SuSiE-inf'

    @classmethod
    def susie(cls) -> tuple[FinemappingMethod, ...]:
        """The methods of the SuSiE family."""
        return (cls.SUSIE, cls.SUSIE_INF)


class Chromosome(StrEnum):
    """Chromosomes a credible set or a variant may lie on, in genome order.

    These are the Ensembl canonical chromosomes, which the genes and the gnomAD variants that
    credible sets are joined to lie on. There is no XY: a pseudoautosomal variant is on X or Y.
    """

    CHR_1 = '1'
    CHR_2 = '2'
    CHR_3 = '3'
    CHR_4 = '4'
    CHR_5 = '5'
    CHR_6 = '6'
    CHR_7 = '7'
    CHR_8 = '8'
    CHR_9 = '9'
    CHR_10 = '10'
    CHR_11 = '11'
    CHR_12 = '12'
    CHR_13 = '13'
    CHR_14 = '14'
    CHR_15 = '15'
    CHR_16 = '16'
    CHR_17 = '17'
    CHR_18 = '18'
    CHR_19 = '19'
    CHR_20 = '20'
    CHR_21 = '21'
    CHR_22 = '22'
    CHR_X = 'X'
    CHR_Y = 'Y'
    CHR_MT = 'MT'


class GenomicRegion(NamedTuple):
    """A region on one chromosome, with inclusive bounds."""

    chromosome: Chromosome
    start: int
    end: int


class KnownGenomicRegion(Enum):
    """Known genomic regions."""

    MHC = GenomicRegion(Chromosome.CHR_6, 25726063, 33400556)
