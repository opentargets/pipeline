"""Variant: the `variant` output.

Generated from the 26.12 release output (8,518,109 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules, then
adjusted by hand, checked against the same release:

* `variantId` is `{chromosome}_{position}_{ref}_{alt}` (underscores, not the dashes croissant's description
  shows), or, for a variant too long to spell out, `OTVAR_{chromosome}_{position}_{md5}` (2,114 rows).
* Alleles are bases only (`[ACGTN]+`); chromosomes are 1-22, X, Y and MT.
* `variantEffect[].normalisedScore` is bounded to [-1, 1], as its description says; the raw `score` is
  method-specific and unbounded.
* `variantEffect[].assessmentFlag` holds LOFTEE flags, comma-separated: its shape is checked, not the
  combinations seen, since LOFTEE has more flags than one release shows.
* `alleleFrequencies[].populationName` is a gnomAD adjusted (`_adj`) ancestry group: its shape is checked,
  not the groups, which gnomAD changes between versions (it only looks like an ontology id by accident).
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ENSEMBL_TRANSCRIPT_ID, ONTOLOGY_ID, RS_ID

#: `{chromosome}_{position}_{ref}_{alt}`, or `OTVAR_{chromosome}_{position}_{md5}` for a long variant.
VARIANT_ID = r'^((\d+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+|OTVAR_(\d+|X|Y|MT)_\d+_[0-9a-f]{32})$'

#: An allele: bases only.
ALLELE = r'^[ACGTN]+$'

#: Comma-separated LOFTEE flags.
LOFTEE_FLAGS = r'^[0-9A-Z_]+(,[0-9A-Z_]+)*$'

Chromosome = Literal[
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12',
    '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', 'X', 'Y', 'MT',
]  # fmt: skip

#: A gnomAD genetic ancestry group's adjusted (`_adj`) frequency: the groups themselves are gnomAD's to change.
GNOMAD_POPULATION = r'^[a-z]+_adj$'


class VariantEffect(StructModel):
    """`variantEffect[]`."""

    method: Literal['AlphaMissense', 'FoldX', 'GERP', 'GPN-Star', 'LOFTEE', 'SIFT', 'VEP'] = Field(
        description='Method name used to predict the effect of the variant'
    )
    assessment: str | None = Field(min_length=1, description='Textual assessment of the variant effect')
    score: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(description='Score of the variant effect')
    assessmentFlag: str | None = Field(
        pattern=LOFTEE_FLAGS, description='Flagging if the variant effect is considered pathogenic'
    )
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] | None = Field(
        pattern=ENSEMBL_GENE_ID, description='Target identifier on which the variant effect is interpreted'
    )
    normalisedScore: float = Field(ge=-1.0, le=1.0, description='Variant effect normalised between -1 and 1')


class TranscriptConsequences(StructModel):
    """`transcriptConsequences[]`."""

    variantFunctionalConsequenceIds: list[
        Annotated[Annotated[str, Bioregistry('so'), Reference('so', 'id')], Field(pattern=ONTOLOGY_ID)]
    ] = Field(
        description=(
            'The sequence ontology identifier of the consequence of the variant based on Ensembl VEP in the '
            'context of the transcript'
        )
    )
    aminoAcidChange: str | None = Field(
        min_length=1, description='Amino acid change caused by this variant on this gene'
    )
    uniprotAccessions: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] = Field(
        description='Uniprot identifiers of the gene product'
    )
    isEnsemblCanonical: bool = Field(description='Flagging if the transcript is the canonical transcript for the gene')
    codons: str | None = Field(min_length=1, description='Affected codon in the transcript')
    distanceFromFootprint: int = Field(ge=0, description='Distance of the variant from the transcript')
    distanceFromTss: int = Field(description='Distance of the variant from the transcription start site')
    appris: str | None = Field(min_length=1, description='Appris annotation of the transcript')
    maneSelect: str | None = Field(min_length=1, description='MANE annotation of the transcript')
    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Target target identifier of the transcript'
    )
    impact: Literal['HIGH', 'LOW', 'MODERATE', 'MODIFIER'] = Field(
        description='Ensembl VEP predicted impact of the variant on the transcript'
    )
    lofteePrediction: Literal['HC', 'LC'] | None = Field(description='Loss-of-function prediction based on LOFTEE')
    consequenceScore: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=0.0, le=1.0, description='Score assigned to transcript based on Ensembl VEP consequence'
    )
    transcriptIndex: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=0, description='Index of the transcript in the list of transcripts around the gene'
    )
    approvedSymbol: str | None = Field(min_length=1, description='HGNC gene symbol')
    biotype: str = Field(min_length=1, description='Biotype of the transcript')
    transcriptId: Annotated[str, Bioregistry('ensembl')] = Field(
        pattern=ENSEMBL_TRANSCRIPT_ID, description='Ensembl transcript identifier'
    )


class AlleleFrequencies(StructModel):
    """`alleleFrequencies[]`."""

    populationName: str = Field(pattern=GNOMAD_POPULATION, description='Name of the population')
    alleleFrequency: float | None = Field(
        ge=0.0, le=1.0, description='Frequency of the alternate allele in the population'
    )


class DbXrefs(StructModel):
    """`dbXrefs[]`."""

    id: str = Field(min_length=1, description='Identifier of the variant in the given database')
    source: Literal['clinvar', 'ensembl_variation', 'gnomad', 'omim', 'protvar'] = Field(
        description='Name of the database the variant is referenced in'
    )


class Variant(DatasetModel):
    """Variant."""

    dataset_name: ClassVar[str] = 'variant'
    dataset_title: ClassVar[str] = 'Variant'
    dataset_description: ClassVar[str] = (
        'Core variant information for all variants in the Platform. Variants are included if any phenotypic '
        'information is available for the variant, including GWAS or molQTL credible sets, ClinVar, Uniprot or '
        'ClinPGx. The dataset includes variant metadata as well as variant effects derived from Ensembl VEP'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'variantId': '1_17005_A_G',
        'chromosome': '1',
        'position': 17005,
        'referenceAllele': 'A',
        'alternateAllele': 'G',
        'variantEffect': [
            {
                'method': 'VEP',
                'assessment': 'non_coding_transcript_exon_variant',
                'score': 0.3400000035762787,
                'assessmentFlag': 'END_TRUNC',
                'targetId': 'ENSG00000227232',
                'normalisedScore': 0.3400000035762787,
            }
        ],
        'mostSevereConsequenceId': 'SO_0001792',
        'transcriptConsequences': [
            {
                'variantFunctionalConsequenceIds': ['SO_0001792'],
                'aminoAcidChange': 'G239E',
                'uniprotAccessions': ['Q6IEY1'],
                'isEnsemblCanonical': True,
                'codons': 'ccA/ccG',
                'distanceFromFootprint': 0,
                'distanceFromTss': -7881,
                'appris': 'P1',
                'maneSelect': 'NM_001005221.2',
                'targetId': 'ENSG00000227232',
                'impact': 'MODIFIER',
                'lofteePrediction': 'HC',
                'consequenceScore': 0.3400000035762787,
                'transcriptIndex': 1,
                'approvedSymbol': 'WASH7P',
                'biotype': 'transcribed_unprocessed_pseudogene',
                'transcriptId': 'ENST00000488147',
            }
        ],
        'rsIds': ['rs201833382'],
        'hgvsId': '1:g.17005A>G',
        'alleleFrequencies': [{'populationName': 'sas_adj', 'alleleFrequency': 0.026576384562415976}],
        'dbXrefs': [{'id': 'rs201833382', 'source': 'ensembl_variation'}],
        'variantDescription': (
            'Non coding transcript exon variant overlapping with WASH7P, a transcribed unprocessed pseudogene '
            'The closest protein-coding gene is OR4F5 (48,414 basepair away).'
        ),
    }

    variantId: Annotated[str, PrimaryKey()] = Field(
        pattern=VARIANT_ID,
        description=(
            'Unique identifier for the variant following schema: '
            '{chromosome}-{position}-{referenceAllele}-{alternateAllele}'
        ),
    )
    chromosome: Chromosome = Field(description='Chromosome on which the variant is located')
    position: Annotated[int, PolarsDtype(pl.Int32)] = Field(ge=0, description="Variant's position on the chromosome")
    referenceAllele: str = Field(pattern=ALLELE, description='Reference allele for the variant')
    alternateAllele: str = Field(pattern=ALLELE, description='Alternate allele for the variant')
    variantEffect: list[VariantEffect] = Field(
        description='Predicted or measured effect of the variant based on various methods'
    )
    mostSevereConsequenceId: Annotated[str, Bioregistry('so'), Reference('so', 'id')] = Field(
        pattern=ONTOLOGY_ID,
        description=('Sequence ontology identifier of the most severe consequence of the variant based on Ensembl VEP'),
    )
    transcriptConsequences: list[TranscriptConsequences] = Field(
        description='Predicted consequences on transcript context'
    )
    rsIds: list[Annotated[str, Field(pattern=RS_ID)]] = Field(description='RsIds for the variant')
    hgvsId: str | None = Field(min_length=1, description='HGVS identifier of the variant')
    alleleFrequencies: list[AlleleFrequencies] = Field(
        description='Allele frequencies of the variant in different populations'
    )
    dbXrefs: list[DbXrefs] = Field(description='Cross-references for the variant in different databases')
    variantDescription: str = Field(min_length=1, description='Short summary of the variant effect')
