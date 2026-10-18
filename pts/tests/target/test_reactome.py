"""The Reactome pathways builder."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from pts.transformers.target.reactome import ENSEMBL2REACTOME_COLUMNS, build_reactome, read_reactome_pathways
from tests.target.rows import rows

LINES = [
    'ENSG00000141510\tR-HSA-69278\turl\tCell Cycle, Mitotic\tTAS\tHomo sapiens',
    'ENSG00000141510\tR-HSA-69278\turl\tCell Cycle, Mitotic\tIEA\tHomo sapiens',
    'ENSG00000141510\tR-HSA-1\thttps://reactome.org\tCell Cycle\tTAS\tHomo sapiens',
    'ENSMUSG0000001\tR-MMU-69278\thttps://reactome.org\tCell Cycle\tIEA\tMus musculus',
    'ENST00000269305\tR-HSA-69278\thttps://reactome.org\tCell Cycle, Mitotic\tIEA\tHomo sapiens',
    'ENSG00000000002\tR-HSA-404\thttps://reactome.org\tNo top level\tIEA\tHomo sapiens',
]
REACTOME = pl.DataFrame(
    [
        ('R-HSA-69278', 'Cell Cycle, Mitotic', [['R-HSA-1', 'R-HSA-69278'], ['R-HSA-2', 'R-HSA-69278']]),
        ('R-HSA-1', 'Cell Cycle', [['R-HSA-1']]),
        ('R-HSA-2', 'Other root', [['R-HSA-2']]),
        ('R-HSA-404', 'Orphan', []),
    ],
    schema={'id': pl.String, 'label': pl.String, 'path': pl.List(pl.List(pl.String))},
    orient='row',
)


def _pathways(tmp_path: Path) -> pl.DataFrame:
    path = tmp_path / 'Ensembl2Reactome.txt'
    path.write_text('\n'.join(LINES) + '\n')
    return read_reactome_pathways(str(path))


def test_read_names_the_columns(tmp_path: Path) -> None:
    assert _pathways(tmp_path).columns == ENSEMBL2REACTOME_COLUMNS


def test_human_genes_with_the_first_path_top_level_term(tmp_path: Path) -> None:
    assert build_reactome(_pathways(tmp_path), REACTOME).to_dicts() == [
        {
            'id': 'ENSG00000141510',
            'pathways': [
                {'pathwayId': 'R-HSA-1', 'pathway': 'Cell Cycle', 'topLevelTerm': 'Cell Cycle'},
                {'pathwayId': 'R-HSA-69278', 'pathway': 'Cell Cycle, Mitotic', 'topLevelTerm': 'Cell Cycle'},
            ],
        }
    ]


def test_matches_the_spark_output(tmp_path: Path) -> None:
    assert rows(build_reactome(_pathways(tmp_path), REACTOME), 'id', ('pathways',)) == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'id': 'ENSG00000141510',
        'pathways': [
            {'pathwayId': 'R-HSA-1', 'pathway': 'Cell Cycle', 'topLevelTerm': 'Cell Cycle'},
            {'pathwayId': 'R-HSA-69278', 'pathway': 'Cell Cycle, Mitotic', 'topLevelTerm': 'Cell Cycle'},
        ],
    }
]
