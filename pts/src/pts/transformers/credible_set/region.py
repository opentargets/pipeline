"""Genomic regions that a rule treats apart from the rest of the genome.

The MHC is one: its long-range LD makes the credible sets whose lead lies in it unreliable, so set
QC flags them IN_MHC.
"""

from __future__ import annotations

from enum import Enum
from typing import NamedTuple

from pts.schemas.credible_set import Chromosome


class GenomicRegion(NamedTuple):
    """A region on one chromosome, with inclusive bounds."""

    chromosome: Chromosome
    start: int
    end: int


class KnownGenomicRegion(Enum):
    """Known genomic regions."""

    MHC = GenomicRegion(Chromosome.CHR_6, 25726063, 33400556)
