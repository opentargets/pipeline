"""Target: the `target` output.

Generated from the 26.12 release output (78,733 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class GenomicLocation(StructModel):
    """`genomicLocation`."""

    chromosome: str = Field(min_length=1, description='Chromosome on which the target located')
    start: int = Field(ge=0, description='Genomic start position of the target gene')
    end: int = Field(ge=0, description='Genomic end position of the target gene')
    strand: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        description=('Strand orientation of the target gene: 1 for the forward strand, -1 for the reverse strand')
    )


class Go(StructModel):
    """`go[]`."""

    id: Annotated[str, Bioregistry('go'), Reference('go', 'id')] = Field(
        min_length=1, description='GO identifier for the annotation'
    )
    source: str = Field(
        min_length=1, description='Source database and identifer from which the ontology term was sourced'
    )
    evidence: str = Field(min_length=1, description='Evidence supporting the GO annotation')
    aspect: Literal['C', 'F', 'P'] = Field(
        description=(
            'Type of the GO annotation: molecular function (F), biological process (P) and cellular localisation (C)'
        )
    )
    geneProduct: Annotated[str, Bioregistry('uniprot')] = Field(
        min_length=1, description='Gene product associated with the GO annotation'
    )
    ecoId: Annotated[str, Bioregistry('eco')] | None = Field(
        min_length=1, description='Evidence & Conclusion Ontology (ECO) identifier for the GO annotation'
    )


class Attributes(StructModel):
    """`hallmarks/attributes[]`."""

    pmid: Annotated[int, Bioregistry('pubmed')] | None = Field(
        ge=0, description='PubMed ID of the supporting literature for the hallmark attribute'
    )
    description: str = Field(min_length=1, description='Description of the hallmark attribute')
    name: str = Field(min_length=1, description='Name of the hallmark attribute')


class CancerHallmarks(StructModel):
    """`hallmarks/cancerHallmarks[]`."""

    pmid: Annotated[int, Bioregistry('pubmed')] = Field(
        ge=0, description='PubMed ID of the supporting literature for the cancer hallmark'
    )
    description: str = Field(min_length=1, description='Description of the cancer hallmark')
    impact: str | None = Field(min_length=1, description='Impact of the cancer hallmark on the target')
    label: str = Field(min_length=1, description='Label associated with the cancer hallmark')


class Hallmarks(StructModel):
    """`hallmarks`."""

    attributes: list[Attributes] | None = Field(description='Attributes of the hallmark annotation')
    cancerHallmarks: list[CancerHallmarks] | None = Field(
        description='Cancer hallmarks associated with the target gene'
    )


class TargetSynonym(StructModel):
    """A synonym of the target: `synonyms[]`, `symbolSynonyms[]` and `nameSynonyms[]`."""

    label: str = Field(min_length=1, description='Synonym label for the target')
    source: Literal['HGNC', 'NCBI_entrez', 'uniprot'] = Field(description='Source database of the synonym')






class SubcellularLocations(StructModel):
    """`subcellularLocations[]`."""

    location: str = Field(min_length=1, description='Name of the subcellular compartment where the protein was found')
    source: Literal['HPA_additional', 'HPA_extracellular_location', 'HPA_main', 'uniprot'] = Field(
        description='Source database for the subcellular location'
    )
    termSL: Annotated[str, Bioregistry('sl')] | None = Field(
        min_length=1, description='Subcellular location term identifier from SwissProt'
    )
    labelSL: str | None = Field(min_length=1, description='Subcellular location category from SwissProt')
    targetModifier: str | None = Field(
        min_length=1, description=('Protein isoform or modification that specific for the given subcellular location')
    )


class TargetClass(StructModel):
    """`targetClass[]`."""

    id: int = Field(ge=0, description='Unique identifier for the target class')
    label: str = Field(min_length=1, description='Label for the target class')
    level: str = Field(min_length=1, description='Hierarchical level of the target class')


class ObsoleteLabel(StructModel):
    """A symbol or name previously used for the target: `obsoleteSymbols[]` and `obsoleteNames[]`."""

    label: str = Field(min_length=1, description='Obsolete symbol or name of the target')
    source: Literal['HGNC'] = Field(description='Source of the obsolete symbol or name')




class Constraint(StructModel):
    """`constraint[]`."""

    constraintType: Literal['lof', 'mis', 'syn'] = Field(description='Type of constraint applied to the target')
    score: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        description='Constraint score indicating gene intolerance'
    )
    exp: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(ge=0.0, description='Expected constraint score')
    obs: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=0, description='Observed constraint score')
    oe: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Observed/Expected (OE) constraint score'
    )
    oeLower: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Lower bound of the OE constraint score'
    )
    oeUpper: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Upper bound of the OE constraint score'
    )
    upperRank: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Rank of the target gene in constraint analysis'
    )
    upperBin: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Binned category for the constraint score'
    )
    upperBin6: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Six-bin classification of the constraint score'
    )


class ProteinIds(StructModel):
    """`proteinIds[]`."""

    id: str = Field(min_length=1, description='Protein ID for the target')
    source: str = Field(pattern=ONTOLOGY_ID, description='Source database for the protein ID')


class DbXrefs(StructModel):
    """`dbXrefs[]`."""

    id: str = Field(min_length=1, description='Cross-reference ID for external databases')
    source: Literal['ChEMBL', 'DrugBank', 'HGNC', 'InterPro', 'PDB', 'RNAcentral', 'Reactome', 'signalP'] = Field(
        description='Source database for the cross-reference'
    )


class Pathways(StructModel):
    """`pathways[]`."""

    pathwayId: str = Field(min_length=1, description='Unique identifier for the pathway')
    pathway: str = Field(min_length=1, description='Name of the pathway')
    topLevelTerm: str = Field(min_length=1, description='Top-level term categorising the pathway')


class Target(DatasetModel):
    """Target."""

    dataset_name: ClassVar[str] = 'target'
    dataset_title: ClassVar[str] = 'Target'
    dataset_description: ClassVar[str] = (
        'Core annotation for drug targets (gene/proteins). Targets are defined '
        'based on EMBL-EBI Ensembl database and uses the Ensembl gene ID as '
        'the  primary identifier. An Ensembl gene ID is considered potential '
        'drug target if included in the canonical assembly or if present '
        'alternative assemblies but encoding for a reviewed protein product '
        'according to the UniProt database'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'id': 'ENSG00000000003',
        'approvedSymbol': 'TSPAN6',
        'biotype': 'protein_coding',
        'genomicLocation': {'chromosome': 'X', 'start': 100627080, 'end': 100639991, 'strand': -1},
        'approvedName': 'tetraspanin 6',
        'go': [
            {
                'id': 'GO:0005515',
                'source': 'PMID:22908223',
                'evidence': 'IPI',
                'aspect': 'F',
                'geneProduct': 'O43657',
                'ecoId': 'ECO:0000353',
            }
        ],
        'hallmarks': {
            'attributes': [
                {
                    'pmid': 12050232,
                    'description': (
                        'essential regulator of embryonic uterine development and the cyclic '
                        'development of the adult uterine endometrium'
                    ),
                    'name': 'differentiation and development',
                }
            ],
            'cancerHallmarks': [
                {
                    'pmid': 19372592,
                    'description': 'promotes fibroblast proliferation and represses expression of p53',
                    'impact': 'promotes',
                    'label': 'proliferative signalling',
                }
            ],
        },
        'synonyms': [{'label': 'Tetraspanin-6', 'source': 'uniprot'}],
        'symbolSynonyms': [{'label': 'Tspan-6', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'Tetraspanin-6', 'source': 'uniprot'}],
        'functionDescriptions': ['May be an angiogenesis inhibitor.'],
        'subcellularLocations': [
            {
                'location': 'Membrane; Multi-pass membrane protein',
                'source': 'uniprot',
                'termSL': 'SL-0162',
                'labelSL': 'Cellular component',
                'targetModifier': 'Isoform 1',
            }
        ],
        'targetClass': [{'id': 643, 'label': 'Enzyme', 'level': 'l1'}],
        'obsoleteSymbols': [{'label': 'TM4SF6', 'source': 'HGNC'}],
        'obsoleteNames': [{'label': 'transmembrane 4 superfamily member 6', 'source': 'HGNC'}],
        'constraint': [
            {
                'constraintType': 'syn',
                'score': -1.091599941253662,
                'exp': 67.02200317382812,
                'obs': 82,
                'oe': 1.2235000133514404,
                'oeLower': 1.0235999822616577,
                'oeUpper': 1.4701000452041626,
                'upperRank': 13048,
                'upperBin': 7,
                'upperBin6': 4,
            }
        ],
        'proteinIds': [{'id': 'O43657', 'source': 'uniprot_swissprot'}],
        'dbXrefs': [{'id': '11858', 'source': 'HGNC'}],
        'pathways': [
            {
                'pathwayId': 'R-HSA-162699',
                'pathway': 'Synthesis of dolichyl-phosphate mannose',
                'topLevelTerm': 'Metabolism of proteins',
            }
        ],
        'tss': 100636806,
    }

    id: Annotated[str, Bioregistry('ensembl'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Unique identifier for the target'
    )
    approvedSymbol: str = Field(min_length=1, description='Approved gene symbol of the target')
    biotype: str = Field(
        description=('Biotype classification of the target gene, indicating if the gene is protein coding')
    )
    genomicLocation: GenomicLocation = Field(description='Genomic location information of the target gene')
    approvedName: str = Field(description='Approved full name of the target gene')
    go: list[Go] | None = Field(description='List of Gene Ontology (GO) annotations related to the target')
    hallmarks: Hallmarks | None = Field(description='Hallmarks related to the target gene sourced from COSMIC')
    synonyms: list[TargetSynonym] = Field(description='List of synonyms for the target gene')
    symbolSynonyms: list[TargetSynonym] = Field(description='List of symbol-based synonyms for the target gene')
    nameSynonyms: list[TargetSynonym] = Field(description='List of name-based synonyms for the target gene')
    functionDescriptions: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Functional descriptions of the target gene sourced from UniProt'
    )
    subcellularLocations: list[SubcellularLocations] | None = Field(
        description='List of subcellular locations where the target protein is found'
    )
    targetClass: list[TargetClass] | None = Field(description='Target classification categories from ChEMBL')
    obsoleteSymbols: list[ObsoleteLabel] = Field(
        description='List of obsolete symbols previously used for the target gene'
    )
    obsoleteNames: list[ObsoleteLabel] = Field(description='List of obsolete names previously used for the target gene')
    constraint: list[Constraint] | None = Field(description='Constraint scores for the target gene from gnomAD')
    proteinIds: list[ProteinIds] | None = Field(description='Protein identifiers associated with the target')
    dbXrefs: list[DbXrefs] = Field(description='Database cross-references for the target')
    pathways: list[Pathways] | None = Field(description='Pathway annotations for the target')
    tss: int = Field(ge=0, description='Transcription Start Site (TSS) information for the target')
