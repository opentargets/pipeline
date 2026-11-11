"""Protein-function annotations of a variant for one gene, from the variant index.

VIDRA scores protein function on `[0, 1]`, 1 benign and 0 damaging, with two predictors read from
`variantEffect` of the variant index: GPN-Star (genome-wide, SNVs only; see `gpn_star`) and
AlphaMissense (missense only, per gene). The consequence of the variant on the gene's canonical
transcript decides how a missing predictor is imputed downstream, through `consequence_class`:

- `lof`: the variant is hard-coded maximally damaging (both scores 0);
- `missense`: a missing score takes the gene's mean over its missense variants;
- `inframe_or_splice`: in-frame indels and splice-region changes also take the gene mean;
- `other`: synonymous, UTR, intronic and non-coding changes are benign for AlphaMissense and keep
  the observed GPN-Star score;
- `unknown`: the variant is not in the index for that gene, so nothing is known about it and the
  gene mean applies, as in the original pipeline.
"""

from __future__ import annotations

LOF_CONSEQUENCES = frozenset({
    'SO_0001587',  # stop_gained
    'SO_0002012',  # start_lost
    'SO_0001589',  # frameshift_variant
    'SO_0001575',  # splice_donor_variant
    'SO_0001574',  # splice_acceptor_variant
})
"""Loss of function: protein-function scores are hard-coded to 0 (maximally damaging)."""

MISSENSE_CONSEQUENCE = 'SO_0001583'

INFRAME_OR_SPLICE_CONSEQUENCES = frozenset({
    'SO_0001821',  # inframe_insertion
    'SO_0001822',  # inframe_deletion
    'SO_0001818',  # protein_altering_variant
    'SO_0001578',  # stop_lost
    'SO_0001630',  # splice_region_variant
    'SO_0002170',  # splice_donor_region_variant
    'SO_0002169',  # splice_polypyrimidine_tract_variant
    'SO_0001626',  # incomplete_terminal_codon_variant
    'SO_0001580',  # coding_sequence_variant
})

CODING_CONSEQUENCES = (
    LOF_CONSEQUENCES
    | {MISSENSE_CONSEQUENCE}
    | INFRAME_OR_SPLICE_CONSEQUENCES
    | {
        'SO_0001819',  # synonymous_variant
        'SO_0001567',  # stop_retained_variant
    }
)
"""What makes a GWAS lead a coding variant of a gene (the original pipeline's `CODING_CONSEQUENCES`)."""

LOFTEE_HIGH_CONFIDENCE = 'HC'

CLASS_LOF = 'lof'
CLASS_MISSENSE = 'missense'
CLASS_INFRAME_OR_SPLICE = 'inframe_or_splice'
CLASS_OTHER = 'other'
CLASS_UNKNOWN = 'unknown'
CONSEQUENCE_CLASSES = (CLASS_LOF, CLASS_MISSENSE, CLASS_INFRAME_OR_SPLICE, CLASS_OTHER, CLASS_UNKNOWN)


def consequence_class(consequences: list[str] | None, loftee: str | None = None) -> str:
    """The imputation class of a variant's consequences on one transcript.

    Examples:
        >>> consequence_class(['SO_0001583'])
        'missense'
        >>> consequence_class(['SO_0001630', 'SO_0001583'])
        'missense'
        >>> consequence_class(['SO_0001589', 'SO_0001583'])
        'lof'
        >>> consequence_class(['SO_0001819'], loftee='HC')
        'lof'
        >>> consequence_class(['SO_0001821'])
        'inframe_or_splice'
        >>> consequence_class(['SO_0001627'])
        'other'
        >>> consequence_class(None)
        'unknown'
    """
    if not consequences:
        return CLASS_UNKNOWN
    terms = set(consequences)
    if loftee == LOFTEE_HIGH_CONFIDENCE or terms & LOF_CONSEQUENCES:
        return CLASS_LOF
    if MISSENSE_CONSEQUENCE in terms:
        return CLASS_MISSENSE
    if terms & INFRAME_OR_SPLICE_CONSEQUENCES:
        return CLASS_INFRAME_OR_SPLICE
    return CLASS_OTHER
