"""GWAS credible-set locus-to-gene evidence: the `evidence` output of `pts_gwas_evidence`.

Descriptions are copied verbatim from croissant's `evidence_gwas_credible_sets.json`, as are its
`isPrimaryKey`/`foreign_key` tags (as `PrimaryKey`/`Reference`).

Rules inferred from the transformer (`pts.transformers.evidence.gwas_evidence`) rather than
confirmed independently against production data:

* `curationDate` uses the transformer's own date regex, unanchored (`str.contains`), so the schema
  accepts exactly what the transformer lets through -- including its "9999-99-99 passes" quirk.
* `publicationDate` comes from the literature look-up table's `firstPublicationDate` instead, and
  is asserted to be a whole ISO date.
* `targetFromSourceId` is an Ensembl gene ID for this datasource (gentropy's L2G `geneId`), even
  though the shared evidence description also allows UniProt IDs and gene symbols.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Note, Reference
from pts.schemas.outputs.evidence_base import ENSEMBL_GENE_ID, ISO_DATE, EvidenceBase

#: The transformer's own curation-date regex (`gwas_evidence.py::_CURATION_DATE_REGEX`).
_LOOSE_DATE = r'\d{4}-\d{2}-\d{2}'


class GwasCredibleSetEvidence(EvidenceBase):
    """GWAS credible-set locus-to-gene evidence."""

    dataset_name: ClassVar[str] = 'evidence_gwas_credible_sets'
    dataset_description: ClassVar[str] = 'GWAS credible-set locus-to-gene evidence'
    example: ClassVar[dict] = {
        'diseaseFromSourceMappedId': 'EFO_0004340',
        'diseaseId': 'EFO_0004340',
        'targetId': 'ENSG00000169174',
        'id': '3f2b1c9e8d7a6b5c4d3e2f1a0b9c8d7e6f5a4b3c',
        # valid evidence never carries a flag; one is here so the element rules get exercised
        'qualityControls': ['Duplicated'],
        'score': 0.87,
        'datatypeId': 'genetic_association',
        'datasourceId': 'gwas_credible_sets',
        'targetFromSourceId': 'ENSG00000169174',
        'resourceScore': 0.87,
        'curationDate': '2019-10-04',
        'studyLocusId': '0a1b2c3d4e5f60718293a4b5c6d7e8f9',
        'literature': ['31578528'],
        'publicationDate': '2019-10-04',
        'evidenceDate': '2019-10-04',
    }

    datatypeId: Annotated[
        Literal['genetic_association'],
        Note('constant for this datasource; not an enum in the shared evidence spec'),
    ] = Field(description='Type of the evidence')
    datasourceId: Literal['gwas_credible_sets'] = Field(description='Identifer of the evidence source')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl gene ID, Uniprot '
            'ID, gene symbol), only capital letters are accepted'
        ),
    )
    resourceScore: float = Field(
        ge=0.0,
        le=1.0,
        description='Score provided by datasource indicating strength of target-disease association',
    )
    curationDate: str | None = Field(pattern=_LOOSE_DATE, description='Date of the GWAS study curation')
    studyLocusId: Annotated[str, Reference('credible_set', 'studyLocusId')] = Field(
        description='Identifier of the Open Targets Study Locus',
    )
    literature: Annotated[
        list[Annotated[str, Field(pattern=r'^\d+$')]] | None,
        Note('digits-only (PMID) for this datasource; other sources may carry PPR/AGR ids'),
    ] = Field(description='List of PubMed or preprint reference identifiers')
    publicationDate: str | None = Field(
        pattern=ISO_DATE,
        description='Date of the earliest publication supporting the evidence',
    )
    evidenceDate: str | None = Field(description='Earliest data for the evidence')
