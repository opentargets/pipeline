"""Target: the `target` output.

Generated from the 26.09 release output (78,733 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID


class GenomicLocation(StructModel):
    """`genomicLocation`."""

    chromosome: str = Field(min_length=1, description='Chromosome on which the target located')
    start: int = Field(ge=577, le=248936581, description='Genomic start position of the target gene')
    end: int = Field(ge=647, le=248937105, description='Genomic end position of the target gene')
    strand: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=-1,
        le=1,
        description=('Strand orientation of the target gene: 1 for the forward strand, -1 for the reverse strand'),
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
        ge=1180975, le=40040987, description='PubMed ID of the supporting literature for the hallmark attribute'
    )
    description: str = Field(min_length=1, description='Description of the hallmark attribute')
    name: Literal[
        'cell division control',
        'clinical impact',
        'differentiation and development',
        'function summary',
        'fusion partner',
        'global regulation of gene expression',
        'impact of mutation on function',
        'interaction with pathogen',
        'mouse model',
        'role in cancer',
        'senescence',
        'types of alteration in cancer',
    ] = Field(description='Name of the hallmark attribute')


class CancerHallmarks(StructModel):
    """`hallmarks/cancerHallmarks[]`."""

    pmid: Annotated[int, Bioregistry('pubmed')] = Field(
        ge=1180975, le=40125273, description='PubMed ID of the supporting literature for the cancer hallmark'
    )
    description: str = Field(min_length=1, description='Description of the cancer hallmark')
    impact: Literal['promotes', 'promotes ', 'promotes or suppresses', 'suppresses'] | None = Field(
        description='Impact of the cancer hallmark on the target'
    )
    label: Literal[
        'angiogenesis',
        'cell replicative immortality',
        'change of cellular energetics',
        'escaping immune response to cancer',
        'escaping programmed cell death',
        'genome instability and mutations',
        'invasion and metastasis',
        'proliferative signalling',
        'suppression of growth',
        'tumour promoting inflammation',
    ] = Field(description='Label associated with the cancer hallmark')


class Hallmarks(StructModel):
    """`hallmarks`."""

    attributes: list[Attributes] = Field(min_length=1, description='Attributes of the hallmark annotation')
    cancerHallmarks: list[CancerHallmarks] | None = Field(
        min_length=1, description='Cancer hallmarks associated with the target gene'
    )


class Synonyms(StructModel):
    """`synonyms[]`."""

    label: str = Field(min_length=1, description='Synonym label for the target')
    source: Literal['HGNC', 'NCBI_entrez', 'uniprot'] = Field(description='Source database of the synonym')


class SymbolSynonyms(StructModel):
    """`symbolSynonyms[]`."""

    label: str = Field(min_length=1, description='Symbol synonym label for the target')
    source: Literal['HGNC', 'NCBI_entrez', 'uniprot'] = Field(description='Source of the symbol synonym')


class NameSynonyms(StructModel):
    """`nameSynonyms[]`."""

    label: str = Field(min_length=1, description='Name synonym label for the target')
    source: Literal['HGNC', 'NCBI_entrez', 'uniprot'] = Field(description='Source of the name synonym')


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

    id: int = Field(ge=1, le=1715, description='Unique identifier for the target class')
    label: str = Field(min_length=1, description='Label for the target class')
    level: Literal['l1', 'l2', 'l3', 'l4', 'l5', 'l6'] = Field(description='Hierarchical level of the target class')


class ObsoleteSymbols(StructModel):
    """`obsoleteSymbols[]`."""

    label: str = Field(min_length=1, description='Obsolete symbol label')
    source: Literal['HGNC'] = Field(description='Source of the obsolete symbol')


class ObsoleteNames(StructModel):
    """`obsoleteNames[]`."""

    label: str = Field(min_length=1, description='Obsolete name label')
    source: Literal['HGNC'] = Field(description='Source of the obsolete name')


class Constraint(StructModel):
    """`constraint[]`."""

    constraintType: Literal['lof', 'mis', 'syn'] = Field(description='Type of constraint applied to the target')
    score: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=-24.06800079345703, le=17.665000915527344, description='Constraint score indicating gene intolerance'
    )
    exp: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.11238999664783478, le=43424.0, description='Expected constraint score'
    )
    obs: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(ge=1, le=40786, description='Observed constraint score')
    oe: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.003090200014412403, le=11.178999900817871, description='Observed/Expected (OE) constraint score'
    )
    oeLower: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0010980999795719981, le=9.181099891662598, description='Lower bound of the OE constraint score'
    )
    oeUpper: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.014658999629318714, le=42.21099853515625, description='Upper bound of the OE constraint score'
    )
    upperRank: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=18255, description='Rank of the target gene in constraint analysis'
    )
    upperBin: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=9, description='Binned category for the constraint score'
    )
    upperBin6: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, le=5, description='Six-bin classification of the constraint score'
    )


class ProteinIds(StructModel):
    """`proteinIds[]`."""

    id: str = Field(min_length=1, description='Protein ID for the target')
    source: Literal['ensembl_PRO', 'uniprot_obsolete', 'uniprot_swissprot', 'uniprot_trembl'] = Field(
        description='Source database for the protein ID'
    )


class DbXrefs(StructModel):
    """`dbXrefs[]`."""

    id: str = Field(min_length=1, description='Cross-reference ID for external databases')
    source: Literal['ChEMBL', 'DrugBank', 'HGNC', 'InterPro', 'PDB', 'Reactome', 'signalP'] = Field(
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
        'id': 'ENSG00000083454',
        'approvedSymbol': 'P2RX5',
        'biotype': 'protein_coding',
        'genomicLocation': {'chromosome': '17', 'start': 3669763, 'end': 3696398, 'strand': -1},
        'approvedName': 'purinergic receptor P2X 5',
        'go': [
            {
                'id': 'GO:0033198',
                'source': 'GO_REF:0000107',
                'evidence': 'IEA',
                'aspect': 'P',
                'geneProduct': 'Q93086',
                'ecoId': 'ECO:0000256',
            }
        ],
        'hallmarks': {
            'attributes': [
                {
                    'pmid': 26551082,
                    'description': (
                        'homeobox transcription factor, critical for intestinal development, '
                        'implicated in differentiation, proliferation, cell adhesion and '
                        'migration and DNA miss-match repair'
                    ),
                    'name': 'function summary',
                }
            ],
            'cancerHallmarks': [
                {
                    'pmid': 26274893,
                    'description': 'deletion of BMPR1a impairs metastasis',
                    'impact': 'promotes',
                    'label': 'invasion and metastasis',
                }
            ],
        },
        'synonyms': [{'label': 'P2X purinoceptor 5', 'source': 'uniprot'}],
        'symbolSynonyms': [{'label': 'P2X5', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'P2X purinoceptor 5', 'source': 'uniprot'}],
        'functionDescriptions': [
            (
                '[Isoform 6]: ATP-gated nonselective transmembrane cation channel '
                'permeable to potassium, sodium and calcium (PubMed:12761352). Unlike '
                'other P2RX receptors, the P2X5 receptor is also permeable to chloride '
                '(PubMed:12761352). May play a supporting role in the inflammatory '
                'response (By similarity). {ECO:0000250|UniProtKB:Q91VE2, '
                'ECO:0000269|PubMed:12761352}.'
            )
        ],
        'subcellularLocations': [
            {
                'location': 'Cell membrane ; Multi-pass membrane protein',
                'source': 'uniprot',
                'termSL': 'SL-0039',
                'labelSL': 'Cellular component',
                'targetModifier': 'Isoform 6',
            }
        ],
        'targetClass': [{'id': 1047, 'label': 'Ligand-gated ion channel', 'level': 'l2'}],
        'obsoleteSymbols': [{'label': 'LZTR2', 'source': 'HGNC'}],
        'obsoleteNames': [{'label': 'purinergic receptor P2X, ligand-gated ion channel, 5', 'source': 'HGNC'}],
        'constraint': [
            {
                'constraintType': 'syn',
                'score': 0.016201000660657883,
                'exp': 226.41000366210938,
                'obs': 226,
                'oe': 0.998199999332428,
                'oeLower': 0.8957200050354004,
                'oeUpper': 1.1145000457763672,
                'upperRank': 12124,
                'upperBin': 6,
                'upperBin6': 3,
            }
        ],
        'proteinIds': [{'id': 'Q93086', 'source': 'uniprot_swissprot'}],
        'dbXrefs': [{'id': '8536', 'source': 'HGNC'}],
        'pathways': [{'pathwayId': 'R-HSA-418346', 'pathway': 'Platelet homeostasis', 'topLevelTerm': 'Hemostasis'}],
        'tss': 3696155,
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
    go: list[Go] | None = Field(
        min_length=1, description='List of Gene Ontology (GO) annotations related to the target'
    )
    hallmarks: Hallmarks | None = Field(description='Hallmarks related to the target gene sourced from COSMIC')
    synonyms: list[Synonyms] = Field(description='List of synonyms for the target gene')
    symbolSynonyms: list[SymbolSynonyms] = Field(description='List of symbol-based synonyms for the target gene')
    nameSynonyms: list[NameSynonyms] = Field(description='List of name-based synonyms for the target gene')
    functionDescriptions: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='Functional descriptions of the target gene sourced from UniProt'
    )
    subcellularLocations: list[SubcellularLocations] | None = Field(
        description='List of subcellular locations where the target protein is found'
    )
    targetClass: list[TargetClass] | None = Field(
        min_length=1, description='Target classification categories from ChEMBL'
    )
    obsoleteSymbols: list[ObsoleteSymbols] = Field(
        description='List of obsolete symbols previously used for the target gene'
    )
    obsoleteNames: list[ObsoleteNames] = Field(description='List of obsolete names previously used for the target gene')
    constraint: list[Constraint] | None = Field(
        min_length=1, description='Constraint scores for the target gene from gnomAD'
    )
    proteinIds: list[ProteinIds] | None = Field(
        min_length=1, description='Protein identifiers associated with the target'
    )
    dbXrefs: list[DbXrefs] = Field(description='Database cross-references for the target')
    pathways: list[Pathways] | None = Field(min_length=1, description='Pathway annotations for the target')
    tss: int = Field(ge=577, le=248936913, description='Transcription Start Site (TSS) information for the target')
