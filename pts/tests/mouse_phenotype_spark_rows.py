"""Rows the removed pyspark mouse phenotype step produced for the fixtures in test_mouse_phenotype.

Captured from the pyspark `mouse_phenotype` step before its removal, run on those fixtures. OUTPUT and
EXCLUDED are (columns, rows) with the lists of structs sorted by their JSON form and literature sorted.

To regenerate, check out the revision before the step was removed and, in a spark test, run on
`_sources(tmp_path)` the old
`_prepare_impc_datasets_for_mouse_phenotypes(_load_impc_datasets_for_mouse_phenotypes(...))`,
`generate_mouse_phenotypes_dataset` with `_create_mp_classification`, then
`filter_mouse_phenotypes_by_target` against the fixture targets; normalise with `_spark_rows`.
CLASSES is `_create_mp_classification` on the ontology of `test_mp_classes_match_spark`
(the `diamond` fixture plus its extra nodes and edges), collected and sorted as tuples.
"""

CLASSES = [
    ('MP:0000002', 'MP:0000002', 'term 2'),
    ('MP:0000003', 'MP:0000003', 'term 3'),
    ('MP:0000004', 'MP:0000002', 'term 2'),
    ('MP:0000004', 'MP:0000003', 'term 3'),
    ('MP:0000005', 'MP:0000002', 'term 2'),
    ('MP:0000006', 'MP:0000002', 'term 2'),
    ('MP:0000006', 'MP:0000006', 'six'),
    ('MP:0000007', 'MP:0000007', None),
    ('MP:0000008', 'MP:0000008', 'eight'),
    ('MP:0000009', 'MP:0000009', ''),
    ('MP:0000010', 'MP:0000002', 'term 2'),
    ('MP:0000010', 'MP:0000003', 'term 3'),
    ('MP:0000011', 'MP:0000011', 'MP:0000011'),
    ('MP:0000012', 'MP:0000002', 'term 2'),
    ('MP:0000012', 'MP:0000003', 'term 3'),
    ('UBERON_1', 'MP:0000003', 'term 3'),
]

OUTPUT = (
    [
        'targetInModel',
        'targetInModelMgiId',
        'targetInModelEnsemblId',
        'targetFromSourceId',
        'modelPhenotypeId',
        'modelPhenotypeLabel',
        'biologicalModels',
        'modelPhenotypeClasses',
    ],
    [
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:1',
                    'literature': ['111', '222', '333'],
                },
                {
                    'allelicComposition': 'A<tm2>/+',
                    'geneticBackground': 'C57BL/6',
                    'id': 'MGI:2',
                    'literature': ['111', '222'],
                },
            ],
            'modelPhenotypeClasses': [
                {'id': 'MP:0000010', 'label': 'term 10'},
                {'id': 'MP:0000020', 'label': 'term 20'},
            ],
            'modelPhenotypeId': 'MP:0000002',
            'modelPhenotypeLabel': 'two',
            'targetFromSourceId': 'ENSG1',
            'targetInModel': 'A',
            'targetInModelEnsemblId': 'ENSMUSG1',
            'targetInModelMgiId': 'MGI:101',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:1',
                    'literature': ['111', '222', '333'],
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000010', 'label': 'term 10'}],
            'modelPhenotypeId': 'MP:0000003',
            'modelPhenotypeLabel': 'three',
            'targetFromSourceId': 'ENSG1',
            'targetInModel': 'A',
            'targetInModelEnsemblId': 'ENSMUSG1',
            'targetInModelMgiId': 'MGI:101',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:7',
                    'literature': ['444'],
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000020', 'label': 'term 20'}],
            'modelPhenotypeId': 'MP:0000004',
            'modelPhenotypeLabel': 'four again',
            'targetFromSourceId': 'ENSG4',
            'targetInModel': 'D',
            'targetInModelEnsemblId': 'ENSMUSG4',
            'targetInModelMgiId': 'MGI:104',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:7',
                    'literature': ['444'],
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000020', 'label': 'term 20'}],
            'modelPhenotypeId': 'MP:0000004',
            'modelPhenotypeLabel': 'four again',
            'targetFromSourceId': 'ENSG5',
            'targetInModel': 'D',
            'targetInModelEnsemblId': 'ENSMUSG4',
            'targetInModelMgiId': 'MGI:104',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:7',
                    'literature': ['444'],
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000020', 'label': 'term 20'}],
            'modelPhenotypeId': 'MP:0000004',
            'modelPhenotypeLabel': 'four',
            'targetFromSourceId': 'ENSG4',
            'targetInModel': 'D',
            'targetInModelEnsemblId': 'ENSMUSG4',
            'targetInModelMgiId': 'MGI:104',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:7',
                    'literature': ['444'],
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000020', 'label': 'term 20'}],
            'modelPhenotypeId': 'MP:0000004',
            'modelPhenotypeLabel': 'four',
            'targetFromSourceId': 'ENSG5',
            'targetInModel': 'D',
            'targetInModelEnsemblId': 'ENSMUSG4',
            'targetInModelMgiId': 'MGI:104',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'B<em1>/B<em1>',
                    'geneticBackground': 'C57BL/6N',
                    'id': None,
                    'literature': None,
                },
                {
                    'allelicComposition': 'B<tm1>/B<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:1',
                    'literature': ['333'],
                },
            ],
            'modelPhenotypeClasses': [
                {'id': 'MP:0000010', 'label': 'term 10'},
                {'id': 'MP:0000020', 'label': 'term 20'},
            ],
            'modelPhenotypeId': 'MP:0000002',
            'modelPhenotypeLabel': 'two',
            'targetFromSourceId': 'ENSG2',
            'targetInModel': 'B',
            'targetInModelEnsemblId': 'ENSMUSG2',
            'targetInModelMgiId': 'MGI:102',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'B<tm1>/B<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:1',
                    'literature': ['333'],
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000010', 'label': 'term 10'}],
            'modelPhenotypeId': 'MP:0000003',
            'modelPhenotypeLabel': 'three',
            'targetFromSourceId': 'ENSG2',
            'targetInModel': 'B',
            'targetInModelEnsemblId': 'ENSMUSG2',
            'targetInModelMgiId': 'MGI:102',
        },
    ],
)

EXCLUDED = (
    [
        'targetFromSourceId',
        'targetInModel',
        'targetInModelMgiId',
        'targetInModelEnsemblId',
        'modelPhenotypeId',
        'modelPhenotypeLabel',
        'biologicalModels',
        'modelPhenotypeClasses',
    ],
    [
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:6',
                    'literature': None,
                }
            ],
            'modelPhenotypeClasses': [
                {'id': 'MP:0000010', 'label': 'term 10'},
                {'id': 'MP:0000020', 'label': 'term 20'},
            ],
            'modelPhenotypeId': 'MP:0000002',
            'modelPhenotypeLabel': 'two',
            'targetFromSourceId': 'ENSG3',
            'targetInModel': 'C',
            'targetInModelEnsemblId': 'ENSMUSG3',
            'targetInModelMgiId': 'MGI:103',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:6',
                    'literature': None,
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000020', 'label': 'term 20'}],
            'modelPhenotypeId': 'MP:0000004',
            'modelPhenotypeLabel': 'four again',
            'targetFromSourceId': 'ENSG3',
            'targetInModel': 'C',
            'targetInModelEnsemblId': 'ENSMUSG3',
            'targetInModelMgiId': 'MGI:103',
        },
        {
            'biologicalModels': [
                {
                    'allelicComposition': 'A<tm1>/A<tm1>',
                    'geneticBackground': 'involves: 129',
                    'id': 'MGI:6',
                    'literature': None,
                }
            ],
            'modelPhenotypeClasses': [{'id': 'MP:0000020', 'label': 'term 20'}],
            'modelPhenotypeId': 'MP:0000004',
            'modelPhenotypeLabel': 'four',
            'targetFromSourceId': 'ENSG3',
            'targetInModel': 'C',
            'targetInModelEnsemblId': 'ENSMUSG3',
            'targetInModelMgiId': 'MGI:103',
        },
    ],
)
