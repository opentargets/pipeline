"""Target view: the `target_view` view (pts `target_view`).

Every target row with its homologues, tractability, safety liabilities, chemical probes and transcripts
nested back in, then the ids of its credible sets and pharmacogenomics records. Each nested list is null
for a target with no row in its source.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import Bioregistry, StructModel, from_output, row_of
from ot_datamodels.outputs.chemical_probe import ChemicalProbe
from ot_datamodels.outputs.homology import Homology
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_TRANSCRIPT_ID, MD5_HEX
from ot_datamodels.outputs.target import Target
from ot_datamodels.outputs.target_safety_event import TargetSafetyEvent
from ot_datamodels.outputs.target_tractability import TargetTractability
from ot_datamodels.outputs.transcript import Transcript
from ot_datamodels.serving import ClickHouseDictionary, Serving

Homologue = row_of(
    Homology,
    [
        'speciesId',
        'speciesName',
        'homologyType',
        'targetGeneId',
        'isHighConfidence',
        'targetGeneSymbol',
        'queryPercentageIdentity',
        'targetPercentageIdentity',
        'priority',
    ],
    name='Homologue',
)
SafetyLiability = row_of(
    TargetSafetyEvent,
    ['event', 'eventId', 'effects', 'biosamples', 'datasource', 'literature', 'url', 'studies'],
    name='SafetyLiability',
)

TractabilityModality = from_output(TargetTractability, 'modality')
TractabilityCategory = from_output(TargetTractability, 'category')
TractabilityValue = from_output(TargetTractability, 'value')


class Tractability(StructModel):
    """`tractability[]`: a `target_tractability` row, its `category` as `id`."""

    modality: TractabilityModality
    id: TractabilityCategory
    value: TractabilityValue


ProbeTargetFromSourceId = from_output(ChemicalProbe, 'targetFromSourceId')
ProbeId = from_output(ChemicalProbe, 'drugFromSource')
ProbeDrugFromSourceId = from_output(ChemicalProbe, 'drugFromSourceId')
ProbeDrugId = from_output(ChemicalProbe, 'drugId')
ProbeMechanismOfAction = from_output(ChemicalProbe, 'mechanismOfAction')
ProbeOrigin = from_output(ChemicalProbe, 'origin')
ProbeControl = from_output(ChemicalProbe, 'control')
ProbeIsHighQuality = from_output(ChemicalProbe, 'isHighQuality')
ProbesDrugsScore = from_output(ChemicalProbe, 'probesDrugsScore')
ProbeScoreInCells = from_output(ChemicalProbe, 'scoreInCells')
ProbeScoreInOrganisms = from_output(ChemicalProbe, 'scoreInOrganisms')
ProbeUrls = from_output(ChemicalProbe, 'urls')


class ChemicalProbeRow(StructModel):
    """`chemicalProbes[]`: a `chemical_probe` row, its `drugFromSource` as `id`."""

    targetFromSourceId: ProbeTargetFromSourceId
    id: ProbeId
    drugFromSourceId: ProbeDrugFromSourceId
    drugId: ProbeDrugId
    mechanismOfAction: ProbeMechanismOfAction
    origin: ProbeOrigin
    control: ProbeControl
    isHighQuality: ProbeIsHighQuality
    probesDrugsScore: ProbesDrugsScore
    scoreInCells: ProbeScoreInCells
    scoreInOrganisms: ProbeScoreInOrganisms
    urls: ProbeUrls


TranscriptId = from_output(Transcript, 'transcriptId')
TranscriptBiotype = from_output(Transcript, 'biotype')
TranslationId = from_output(Transcript, 'proteinId')
IsEnsemblCanonical = from_output(Transcript, 'isEnsemblCanonical')
TranscriptChromosome = from_output(Transcript, 'chromosome')
TranscriptStart = from_output(Transcript, 'start')
TranscriptEnd = from_output(Transcript, 'end')
TranscriptStrand = from_output(Transcript, 'strand')

_UniprotId = Annotated[str, Bioregistry('uniprot')]
# a list element takes no description, so not `from_output`; tests/test_views.py checks it equals transcriptId
_TranscriptIdElement = Annotated[str, Bioregistry('ensembl'), Field(pattern=ENSEMBL_TRANSCRIPT_ID)]
# list elements, so not `from_output`; tests/test_views.py checks them against credible_set.id and
# pharmacogenomics.pharmacogenomicsId
_CredibleSetId = Annotated[str, Field(pattern=MD5_HEX)]
_PharmacogenomicsId = Annotated[int, Field(ge=0)]


class TranscriptRow(StructModel):
    """`transcripts[]`: a `transcript` row, with the first of each of its UniProt and AlphaFold lists."""

    transcriptId: TranscriptId
    biotype: TranscriptBiotype
    uniprotId: _UniprotId | None = Field(
        min_length=1, description='First SwissProt accession of the transcript, else its first TrEMBL one'
    )
    isUniprotReviewed: bool | None = Field(
        description='Whether uniprotId is a SwissProt (reviewed) accession; null without one'
    )
    translationId: TranslationId
    alphafoldId: str | None = Field(min_length=1, description='First AlphaFold identifier of the transcript')
    uniprotIsoformId: _UniprotId | None = Field(min_length=1, description='First UniProt isoform of the transcript')
    isEnsemblCanonical: IsEnsemblCanonical


class CanonicalTranscript(StructModel):
    """`canonicalTranscript`: the target's Ensembl canonical `transcript` row, its `transcriptId` as `id`."""

    id: TranscriptId
    chromosome: TranscriptChromosome
    start: TranscriptStart
    end: TranscriptEnd
    strand: TranscriptStrand


class TargetView(Target):
    """Target view."""

    dataset_name: ClassVar[str] = 'target_view'
    dataset_title: ClassVar[str] = 'Target view'
    dataset_description: ClassVar[str] = (
        'Targets with their homologues, tractability, safety liabilities, chemical probes and transcripts '
        'nested in, in the shape target had before they were split out of it'
    )
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        dictionaries=(ClickHouseDictionary('target_symbols', key='id', attributes=('approvedSymbol',)),),
        column_types={
            'genomicLocation/start': 'UInt32',
            'genomicLocation/end': 'UInt32',
            'genomicLocation/strand': "Enum8('-1' = -1, '0' = 0, '1' = 1)",
            'hallmarks/attributes[]/pmid': 'UInt32',
            'hallmarks/cancerHallmarks[]/pmid': 'UInt32',
            'targetClass[]/id': 'UInt32',
            'constraint[]/score': 'Float64',
            'constraint[]/exp': 'Float64',
            'constraint[]/obs': 'UInt32',
            'constraint[]/oe': 'Float64',
            'constraint[]/oeLower': 'Float64',
            'constraint[]/oeUpper': 'Float64',
            'constraint[]/upperRank': 'UInt32',
            'constraint[]/upperBin': 'UInt32',
            'constraint[]/upperBin6': 'UInt32',
            'canonicalTranscript/start': 'UInt32',
            'canonicalTranscript/end': 'UInt32',
            'canonicalTranscript/strand': "Enum8('-1' = -1, '0' = 0, '1' = 1)",
            'pharmacogenomicsIds[]': 'UInt64',
        },
    )
    example: ClassVar[dict] = Target.example | {
        'homologues': [{k: v for k, v in Homology.example.items() if k != 'targetId'}],
        'tractability': [{'modality': 'SM', 'id': 'Approved_Drug', 'value': False}],
        'safetyLiabilities': None,
        'chemicalProbes': None,
        'transcriptIds': ['ENST00000373020'],
        'transcripts': [
            {
                'transcriptId': 'ENST00000373020',
                'biotype': 'protein_coding',
                'uniprotId': 'O43657',
                'isUniprotReviewed': True,
                'translationId': 'ENSP00000362111',
                'alphafoldId': 'AF-O43657-F1',
                'uniprotIsoformId': 'Q9H2S6-1',
                'isEnsemblCanonical': True,
            }
        ],
        'canonicalTranscript': {
            'id': 'ENST00000373020',
            'chromosome': 'X',
            'start': 100627108,
            'end': 100636806,
            'strand': -1,
        },
        'canonicalExons': ['100636608', '100636806'],
        'alternativeGenes': None,
        'credibleSetIds': ['0002c80bb07a1c4ab11e5b6d2c9e0d6c'],
        'pharmacogenomicsIds': [1],
    }

    homologues: list[Homologue] | None = Field(description='Homologues of the target, by ascending priority')
    tractability: list[Tractability] | None = Field(description='Tractability assessments of the target')
    safetyLiabilities: list[SafetyLiability] | None = Field(description='Safety events of the target')
    chemicalProbes: list[ChemicalProbeRow] | None = Field(description='Chemical probes of the target')
    transcriptIds: list[_TranscriptIdElement] | None = Field(description='Transcripts of the target, sorted')
    transcripts: list[TranscriptRow] | None = Field(description='Transcripts of the target')
    canonicalTranscript: CanonicalTranscript = Field(description='Ensembl canonical transcript of the target')
    canonicalExons: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='Start and end of each exon of the canonical transcript, as strings, by start then end'
    )
    alternativeGenes: list[str] | None = Field(description='Always null; kept for the shape of the target dataset')
    credibleSetIds: list[_CredibleSetId] = Field(
        description='Credible sets of the studies whose geneId is the target, distinct and sorted'
    )
    pharmacogenomicsIds: list[_PharmacogenomicsId] = Field(
        description='pharmacogenomicsId of the pharmacogenomics records naming the target, ascending'
    )
