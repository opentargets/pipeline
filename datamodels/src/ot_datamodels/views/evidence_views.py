"""The evidence views, generated per source from the `evidence_*` outputs (pts `copy_view`).

For each output `evidence_<source>`:

* `evidence_<source>_view`, the whole output (a subclass of it), RocksDB on `id`;
* `evidence_<source>_by_disease_and_target_view`, `datasourceId, diseaseId, targetId, score, id` of its rows
  with a disease and a target, MergeTree on `(diseaseId, targetId, score)`;
* `evidence_<source>_by_variant_view`, `datasourceId, variantId, score, id` of its rows with a variant, MergeTree
  on `(variantId, score)`; only for an output with a `variantId` column.

Each table holds one source, so `datasourceId` is in no sort key. Each class is bound to a module global of its
own name, which discovery finds; the classes are listed in `EVIDENCE_VIEWS`.
"""

from collections.abc import Mapping, Sequence
from typing import Any

from ot_datamodels.base import DatasetModel
from ot_datamodels.derive import from_output
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.serving import Serving

#: the ClickHouse type of every evidence output's disease and target ids
_ID_TYPES = {'diseaseId': 'LowCardinality(String)', 'targetId': 'LowCardinality(String)'}

#: the other ClickHouse types of each output's values, where they differ from the dtype's
_OUTPUT_TYPES: Mapping[str, Mapping[str, str]] = {
    'evidence_cancer_gene_census': {
        'mutatedSamples[]/numberMutatedSamples': 'UInt32',
        'mutatedSamples[]/numberSamplesTested': 'UInt32',
        'mutatedSamples[]/numberSamplesWithMutationType': 'UInt32',
    },
    'evidence_europepmc': {
        'textMiningSentences[]/dEnd': 'UInt32',
        'textMiningSentences[]/dStart': 'UInt32',
        'textMiningSentences[]/tEnd': 'UInt32',
        'textMiningSentences[]/tStart': 'UInt32',
        'publicationYear': 'UInt16',
    },
    'evidence_expression_atlas': {'log2FoldChangePercentileRank': 'Int32'},
    'evidence_gene_burden': {
        'pValueExponent': 'Int32',
        'studySampleSize': 'UInt32',
        'studyCases': 'UInt32',
        'studyCasesWithQualifyingVariants': 'UInt32',
    },
    'evidence_intogen': {
        'mutatedSamples[]/numberMutatedSamples': 'UInt32',
        'mutatedSamples[]/numberSamplesTested': 'UInt32',
        'mutatedSamples[]/numberSamplesWithMutationType': 'UInt32',
    },
}

#: index view suffix -> (its columns, the columns its rows are non-null in, its sort key, the title's key)
INDEXES: Mapping[str, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], str]] = {
    'by_disease_and_target': (
        ('datasourceId', 'diseaseId', 'targetId', 'score', 'id'),
        ('diseaseId', 'targetId'),
        ('diseaseId', 'targetId', 'score'),
        'disease and target',
    ),
    'by_variant': (
        ('datasourceId', 'variantId', 'score', 'id'),
        ('variantId',),
        ('variantId', 'score'),
        'variant',
    ),
}


def _class_name(dataset_name: str) -> str:
    return ''.join(part.capitalize() for part in dataset_name.split('_'))


def _types(output: str, columns: Sequence[str] | None = None) -> dict[str, str]:
    """The output's ClickHouse types, of the paths under `columns` (every path when None)."""
    types = {**_OUTPUT_TYPES.get(output, {}), **_ID_TYPES}
    if columns is None:
        return types
    return {path: t for path, t in types.items() if path.split('/')[0].removesuffix('[]') in columns}


def _model(name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> type[DatasetModel]:
    cls = type(bases[0])(_class_name(name), bases, {'__module__': __name__, 'dataset_name': name, **namespace})
    assert issubclass(cls, DatasetModel)
    globals()[cls.__name__] = cls
    return cls


def _whole(output: type[DatasetModel]) -> type[DatasetModel]:
    name = f'{output.dataset_name}_view'
    return _model(
        name,
        (output,),
        {
            '__doc__': f'{output.dataset_title} view.',
            'dataset_title': f'{output.dataset_title} view',
            'dataset_description': output.dataset_description,
            'clickhouse': Serving(engine='rocksdb', column_types=_types(output.dataset_name)),
        },
    )


def _index(output: type[DatasetModel], suffix: str) -> type[DatasetModel]:
    columns, keys, order_by, key_title = INDEXES[suffix]
    name = f'{output.dataset_name}_{suffix}_view'
    annotations = {
        c: from_output(output, c, nullable=False if c in keys else None, primary_key=c == 'id') for c in columns
    }
    return _model(
        name,
        (DatasetModel,),
        {
            '__doc__': f'{output.dataset_title} by {key_title}.',
            '__annotations__': annotations,
            'dataset_title': f'{output.dataset_title} by {key_title}',
            'dataset_description': (
                f'The evidence ids and scores of `{output.dataset_name}` with a {key_title}, as served by {key_title}'
            ),
            'dataset_tags': output.dataset_tags,
            'products': output.products,
            'example': {c: output.example[c] for c in columns},
            'clickhouse': Serving(
                engine='mergetree', order_by=order_by, column_types=_types(output.dataset_name, columns)
            ),
        },
    )


def _views() -> list[type[DatasetModel]]:
    views = []
    for name, output in sorted(discover_datasets().items()):
        if not name.startswith('evidence_'):
            continue
        views.append(_whole(output))
        views += [_index(output, suffix) for suffix in INDEXES if set(INDEXES[suffix][0]) <= set(output.model_fields)]
    return views


#: every evidence view, by source then whole copy, by disease and target, by variant
EVIDENCE_VIEWS: tuple[type[DatasetModel], ...] = tuple(_views())
