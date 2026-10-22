"""Drug/Clinical candidates: the `drug_molecule` output.

Generated from the 26.09 release output (19,179 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, StructModel
from pts.schemas.outputs.patterns import CHEMBL_ID


class Synonyms(StructModel):
    """`synonyms[]`."""

    label: str = Field(min_length=1, description='The alternative name (synonym) for the drug')
    source: Literal['AACT', 'ChEMBL'] = Field(
        description=('Source of the synonym (e.g. ChEMBL, or AACT for names mined from clinical trials)')
    )


class TradeNames(StructModel):
    """`tradeNames[]`."""

    label: str = Field(min_length=1, description='The brand (trade) name for the drug')
    source: Literal['ChEMBL'] = Field(description='Source of the trade name (e.g. ChEMBL)')


class CrossReferences(StructModel):
    """`crossReferences[]`."""

    source: Literal['Probes&Drugs', 'drugbank'] = Field(description='Database source name')
    ids: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, description='List of identifiers in other databases'
    )


class DrugMolecule(DatasetModel):
    """Drug/Clinical candidates."""

    dataset_name: ClassVar[str] = 'drug_molecule'
    dataset_title: ClassVar[str] = 'Drug/Clinical candidates'
    dataset_description: ClassVar[str] = (
        'Core annotation for drug or clinical candidate molecules. A drug in '
        'the Platform is understood as any bioactive molecule with drug-like '
        'properties included in the EMBL-EBI ChEMBL database. All ChEMBL '
        'molecules fullfilling any of the next criteria are included in the '
        'database: a) molecules with a known indication, b) molecules with a '
        'known mechanism of action, c) ChEMBL molecules included in the '
        'DrugBank database, d) Molecules that are acknowledged as chemical '
        'probes'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug',)
    example: ClassVar[dict] = {
        'id': 'CHEMBL100',
        'canonicalSmiles': 'CC1(C)Oc2ccc(C#N)cc2[C@@H](N2CCCC2=O)[C@@H]1O',
        'inchiKey': 'TVZCRIROJQEVOT-CABCVRRESA-N',
        'molblock': (
            '\n     RDKit          2D\n\n 21 23  0  0  1  0  0  0  0  0999 V2000\n    '
            '1.6500   -0.2004    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '0.5247    0.4916    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '2.1432   -1.4070    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '2.8122    0.4488    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '2.8589    1.7104    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '1.7480    2.4519    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '0.5748    1.8449    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    '
            '1.2569   -2.3749    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            '-4.1833   -0.6125    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            '-0.6485   -0.1155    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            '-2.9833    0.0000    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 0.2159   -2.3364    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            '-0.5505    2.5369    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            '-1.7721    0.6182    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 4.2612    0.3952    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 3.3352   -1.9847    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            '-1.7255    1.8798    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 4.2323    1.6430    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 3.3208    3.0401    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 1.8054   -3.4417    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n   '
            ' 3.1024   -3.2061    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n  '
            '2  1  1  0\n  1  3  1  6\n  4  1  1  0\n  5  4  1  0\n  6  5  1  0\n  7  2 '
            ' 2  0\n  8  3  1  0\n  9 11  3  0\n 10  2  1  0\n 11 14  1  0\n 12  8  2  '
            '0\n 13  7  1  0\n 14 10  2  0\n  4 15  1  1\n 16  3  1  0\n 17 14  1  0\n '
            '18  5  1  0\n 19  5  1  0\n 20  8  1  0\n 21 16  1  0\n  7  6  1  0\n 20 '
            '21  1  0\n 13 17  2  0\nM  END\n'
        ),
        'drugType': 'Small molecule',
        'name': 'LEVCROMAKALIM',
        'parentId': 'CHEMBL777',
        'synonyms': [{'label': '(-)-cromakalim', 'source': 'ChEMBL'}],
        'tradeNames': [{'label': 'Clavulanate potassium component of augmentin', 'source': 'ChEMBL'}],
        'crossReferences': [{'source': 'drugbank', 'ids': ['DB21154']}],
        'childChemblIds': ['CHEMBL1201113'],
        'maximumClinicalStage': 'PREAPPROVAL',
        'description': (
            'Small molecule drug with a maximum clinical stage of Preapproval '
            '(across all indications), with 6 investigational indications.'
        ),
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=CHEMBL_ID, description='Open Targets molecule identifier')
    canonicalSmiles: str | None = Field(
        min_length=1, description='Standardised chemical structure representation using SMILES notation'
    )
    inchiKey: str | None = Field(min_length=1, description='Chemical identifier generated from the molecular structure')
    molblock: str | None = Field(
        min_length=1,
        description=(
            'Mol Block is a chemical structure file format that serves as a '
            'connection table, representing molecules through a list of atoms, '
            'bonds, and spatial coordinates'
        ),
    )
    drugType: Literal[
        'Antibody',
        'Antibody drug conjugate',
        'Cell',
        'Enzyme',
        'Gene',
        'Oligonucleotide',
        'Oligosaccharide',
        'Protein',
        'Small molecule',
        'Unknown',
        'Vaccine component',
    ] = Field(description=("Classification of the molecule's therapeutic category or chemical class (e.g. Antibody)"))
    name: str = Field(min_length=1, description='Generic name of the drug molecule')
    parentId: str | None = Field(
        pattern=CHEMBL_ID,
        description=('Open Targets molecule identifier of the parent molecule for derivative compounds'),
    )
    synonyms: list[Synonyms] = Field(description='List of alternative names for the drug')
    tradeNames: list[TradeNames] = Field(description='List of brand names for the drug')
    crossReferences: list[CrossReferences] | None = Field(
        min_length=1, description='List of identifiers in other databases'
    )
    childChemblIds: list[Annotated[str, Field(pattern=CHEMBL_ID)]] | None = Field(
        min_length=1, description='List of Open Targets molecule identifiers of derivative compounds'
    )
    maximumClinicalStage: Literal[
        'APPROVAL',
        'EARLY_PHASE_1',
        'IND',
        'PHASE_1',
        'PHASE_1_2',
        'PHASE_2',
        'PHASE_2_3',
        'PHASE_3',
        'PREAPPROVAL',
        'PRECLINICAL',
        'UNKNOWN',
    ] = Field(
        description=("Display name of the highest clinical stage reached by the drug (e.g. 'approved', 'phase III')")
    )
    description: str = Field(min_length=1, description="Summary of the drug's clinical development")
