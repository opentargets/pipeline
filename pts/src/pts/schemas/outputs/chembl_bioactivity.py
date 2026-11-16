"""Drug - ChEMBL bioactivity: the `chembl_bioactivity` output.

Generated from the 26.12 release output (2,755,480 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference
from pts.schemas.outputs.patterns import CHEMBL_ID, ENSEMBL_GENE_ID


class ChemblBioactivity(DatasetModel):
    """Drug - ChEMBL bioactivity."""

    dataset_name: ClassVar[str] = 'chembl_bioactivity'
    dataset_title: ClassVar[str] = 'Drug - ChEMBL bioactivity'
    dataset_description: ClassVar[str] = (
        'Bioactivity measurements from ChEMBL 37 with a pChEMBL value: '
        'affinity, inhibition and functional potency of molecules against '
        'human protein targets, mapped to Ensembl gene identifiers, one row '
        'per measurement and target. Measurements from homologue assays are '
        'included, flagged by confidenceScore. Targeted protein degradation '
        'measurements are not included'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Drug', 'Target')
    example: ClassVar[dict] = {
        'activityId': 31864,
        'targetId': 'ENSG00000173083',
        'drugFromSourceId': 'CHEMBL324340',
        'drugId': 'CHEMBL15928',
        'pchemblValue': 5.6,
        'standardMeasurementType': 'IC50',
        'standardMeasurementValue': 2500.0,
        'measurementClass': 'inhibition',
        'activityOutcome': 'inconclusive',
        'assayActionType': 'ANTAGONIST',
        'actionTypes': ['ANTAGONIST'],
        'le': 0.26,
        'lle': 1.3,
        'assayId': 'CHEMBL872937',
        'assayType': 'B',
        'assayDescription': 'In vivo inhibitory activity against human Heparanase',
        'assayOrganism': 'Canis lupus familiaris',
        'confidenceScore': 8,
        'variantFromSource': 'Q165A',
        'targetFromSourceId': 'CHEMBL3921',
        'complexFlag': False,
        'documentId': 'CHEMBL1146658',
        'docType': 'PUBLICATION',
        'pubmedId': 15149688,
        'year': 2004,
    }

    activityId: Annotated[int, PrimaryKey()] = Field(
        ge=0,
        description=(
            'ChEMBL activity identifier of the measurement. With targetId, '
            'identifies the row: a measurement on a protein complex has one row '
            'per target it is credited to'
        ),
    )
    targetId: Annotated[str, Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Open Targets target credited with the measurement: the gene of a '
            'binding-site component where ChEMBL records one, otherwise every gene '
            'of the ChEMBL target'
        ),
    )
    drugFromSourceId: str = Field(pattern=CHEMBL_ID, description='ChEMBL identifier of the tested molecule')
    drugId: Annotated[str, Reference('drug_molecule', 'id')] | None = Field(
        pattern=CHEMBL_ID,
        description=(
            'Open Targets drug identifier of the tested molecule or its parent. '
            'Most tested molecules are not drugs or clinical candidates, so most '
            'rows have none'
        ),
    )
    pchemblValue: float = Field(
        ge=0.0,
        description=(
            'Negative base-10 logarithm of the molar half-maximal response, '
            'concentration or potency, comparable across measurement types'
        ),
    )
    standardMeasurementType: Literal['AC50', 'EC50', 'ED50', 'IC50', 'Kd', 'Ki', 'Potency', 'XC50'] = Field(
        description='Standardised type of the measurement (e.g., IC50)'
    )
    standardMeasurementValue: float = Field(ge=0.0, description='Standardised value of the measurement, in nM')
    measurementClass: Literal['affinity', 'functional_potency', 'inhibition'] = Field(
        description=(
            'What the measurement quantifies: affinity (Ki, Kd), inhibition (IC50) '
            'or functional_potency (EC50, AC50, XC50, Potency, ED50)'
        )
    )
    activityOutcome: Literal['active', 'inactive', 'inconclusive'] | None = Field(
        description=(
            'Outcome the depositor recorded for the measurement: active, inactive '
            'or inconclusive. Recorded mostly for PubChem BioAssay screens, and '
            'for a few other datasets (e.g., a proteomics study). A potency fitted '
            'to a curve judged inactive or inconclusive still has a pChEMBL value. '
            'Empty where no outcome is recorded'
        )
    )
    assayActionType: (
        Literal[
            'ACTIVATOR',
            'AGONIST',
            'ALLOSTERIC ANTAGONIST',
            'ANTAGONIST',
            'BINDING AGENT',
            'BLOCKER',
            'DEGRADER',
            'DISRUPTING AGENT',
            'INHIBITOR',
            'INVERSE AGONIST',
            'MODULATOR',
            'NEGATIVE ALLOSTERIC MODULATOR',
            'OPENER',
            'OTHER',
            'PARTIAL AGONIST',
            'POSITIVE ALLOSTERIC MODULATOR',
            'STABILISER',
            'SUBSTRATE',
        ]
        | None
    ) = Field(
        description=(
            'Effect of the molecule on its target in this measurement (e.g., '
            'INHIBITOR), as ChEMBL records it on the activity'
        )
    )
    actionTypes: list[
        Literal[
            'ACTIVATOR',
            'AGONIST',
            'ALLOSTERIC ANTAGONIST',
            'ANTAGONIST',
            'BINDING AGENT',
            'BLOCKER',
            'DEGRADER',
            'INHIBITOR',
            'INVERSE AGONIST',
            'MODULATOR',
            'NEGATIVE ALLOSTERIC MODULATOR',
            'OPENER',
            'PARTIAL AGONIST',
            'POSITIVE ALLOSTERIC MODULATOR',
            'POSITIVE MODULATOR',
            'RELEASING AGENT',
            'STABILISER',
            'SUBSTRATE',
        ]
    ] = Field(
        description=(
            'Action types (e.g., INHIBITOR) of the curated mechanisms of action of '
            'the drug on this target, from the drug mechanism of action dataset. '
            'Empty where the drug has no mechanism of action on it'
        )
    )
    le: float | None = Field(
        ge=0.0,
        description=(
            'Ligand efficiency: binding energy per heavy atom (kcal/mol per heavy '
            'atom), from ChEMBL. Empty where ChEMBL computes none'
        ),
    )
    lle: float | None = Field(
        description=(
            'Lipophilic ligand efficiency: pChEMBL value minus the calculated logP '
            '(ALogP), from ChEMBL. Empty where ChEMBL computes none'
        )
    )
    assayId: str = Field(pattern=CHEMBL_ID, description='ChEMBL assay identifier')
    assayType: Literal['B', 'F', 'U'] = Field(
        description='ChEMBL assay type: B (binding), F (functional) or U (unclassified)'
    )
    assayDescription: str = Field(min_length=1, description='ChEMBL description of the assay')
    assayOrganism: str | None = Field(
        min_length=1,
        description=(
            'Organism of the protein or system the assay used, where ChEMBL '
            'records it (e.g., Rattus norvegicus for a rat homologue)'
        ),
    )
    confidenceScore: Annotated[int, PolarsDtype(pl.Int16)] = Field(
        ge=0,
        description=(
            'ChEMBL confidence (0-9) in the assignment of the assay to its target: '
            '9, 7 and 5 when the assay used the human protein itself; 8, 6 and 4 '
            'when ChEMBL could not confirm that, usually because the assay names '
            'no organism, sometimes because it used another species (see '
            'assayOrganism). Lower scores (e.g., 2, a subcellular fraction) are '
            'rare'
        ),
    )
    variantFromSource: str | None = Field(
        min_length=1,
        description=(
            'Mutation of the target protein used in the assay, as ChEMBL records '
            'it (e.g., V600E). Empty for the wild type or where ChEMBL names no '
            'mutation'
        ),
    )
    targetFromSourceId: str = Field(pattern=CHEMBL_ID, description='ChEMBL target identifier')
    complexFlag: bool = Field(
        description=(
            'Whether the measurement is credited to more than one target, because '
            'the ChEMBL target is a complex or family with no single binding-site '
            'gene, or its protein maps to several genes'
        )
    )
    documentId: str = Field(pattern=CHEMBL_ID, description='ChEMBL identifier of the source document')
    docType: Literal['DATASET', 'PATENT', 'PUBLICATION'] = Field(
        description='Type of the source document (e.g., PUBLICATION, DATASET)'
    )
    pubmedId: int | None = Field(ge=0, description='PubMed identifier of the source document')
    year: Annotated[int, PolarsDtype(pl.Int32)] | None = Field(
        ge=0, description='Publication year of the source document'
    )
