"""Generate a strict-from-data DatasetModel module per profiled dataset.

Usage: python generate.py <output_dir> <profiles_dir> <croissant_assets_dir> <target_package_dir> [dataset ...]

Rules, all inferred from the profiled release:
* dtype and struct field order exactly as written;
* nullable only where a null (or NaN) was seen;
* lists with no empty value get `min_length=1`; lists that are always empty get `max_length=0`;
* numbers get `ge`/`le` at the observed min/max (an infinite side is left open);
* strings with at most MAX_ENUM distinct values (and at least ENUM_RATIO rows per value) become a `Literal`,
  unless they are keys/references; otherwise the most specific ID shape every value matches becomes a
  `pattern`; otherwise, with no empty string seen, `min_length=1`;
* descriptions, primary keys and references come from croissant; a primary key is kept only if the data is
  unique on it.
"""

import datetime
import json
import keyword
import math
import re
import sys
from pathlib import Path
from typing import Any

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))
from profiler import PATTERNS

MAX_ENUM = 20
ENUM_RATIO = 5
#: Shapes too generic to be worth asserting on their own.
SKIP_PATTERNS = {'curie'}
PATTERN_CONSTANTS = {
    'ensembl_gene': 'ENSEMBL_GENE_ID',
    'ensembl_transcript': 'ENSEMBL_TRANSCRIPT_ID',
    'ensembl_protein': 'ENSEMBL_PROTEIN_ID',
    'sha1': 'SHA1_HEX',
    'md5': 'MD5_HEX',
    'chembl': 'CHEMBL_ID',
    'rsid': 'RS_ID',
    'variant': 'VARIANT_ID',
    'variant_or_hash': 'VARIANT_ID_OR_HASH',
    'iso_date': 'ISO_DATE',
    'digits': 'DIGITS',
    'ontology': 'ONTOLOGY_ID',
}

DTYPES = {
    'String': 'str',
    'Boolean': 'bool',
    'Int64': 'int',
    'Float64': 'float',
    'Int8': 'int',
    'Int16': 'int',
    'Int32': 'int',
    'UInt8': 'int',
    'UInt16': 'int',
    'UInt32': 'int',
    'UInt64': 'int',
    'Float32': 'float',
    'Date': 'datetime.date',
}


BIOREGISTRY = re.compile(r'\s*\[bioregistry:([^\]]+)\]')


def lit(value: Any) -> str:
    """`repr`, with long strings split into implicitly concatenated chunks so lines stay within 120."""
    if isinstance(value, str) and len(value) > 70:
        words = re.split(r'(?<= )', value)
        chunks, current = [], ''
        for word in words:
            if current and len(current) + len(word) > 70:
                chunks.append(current)
                current = ''
            current += word
        chunks.append(current)
        return '(' + ' '.join(_ascii(c) for c in chunks) + ')'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{k!r}: {lit(v)}' for k, v in value.items()) + '}'
    if isinstance(value, list):
        return '[' + ', '.join(lit(v) for v in value) + ']'
    return _ascii(value) if isinstance(value, str) else repr(value)


def _ascii(text: str) -> str:
    """`repr` with non-ASCII characters escaped, so data such as a curly quote or a Greek letter passes RUF001."""
    return repr(text).encode('ascii', 'backslashreplace').decode()


def camel(name: str) -> str:
    parts = re.split(r'[^A-Za-z0-9]+', name)
    out = ''.join(p[:1].upper() + p[1:] for p in parts if p)
    return out if out and not out[0].isdigit() else f'S{out}'


class Module:
    def __init__(self, dataset: str, profile: dict, croissant: dict[str, dict], meta: dict, unique_key: bool) -> None:
        self.dataset = dataset
        self.profile = profile
        self.paths = profile['paths']
        self.croissant = croissant
        self.meta = meta
        self.unique_key = unique_key
        self.structs: list[str] = []
        self.struct_names: set[str] = set()
        self.imports_base: set[str] = set()
        self.patterns: set[str] = set()
        self.uses_annotated = False
        self.uses_literal = False
        self.uses_pl = False
        self.uses_datetime = False
        self.issues: list[str] = []

    # croissant ids drop list markers: `a[]/b` is `a/b`
    def meta_for(self, path: str) -> dict:
        return self.croissant.get(path.replace('[]', ''), {})

    def type_for(self, dtype: Any, path: str, *, key: bool) -> tuple[str, list[str]]:
        """(annotation, Field kwargs) for one value; the annotation includes `| None` when nullable."""
        p = self.paths[path]
        nonnull = p['n'] - p['nulls']
        nullable = p['nulls'] > 0 or (p.get('nan') or 0) > 0
        markers: list[str] = []
        kwargs: list[str] = []
        ref = self.meta_for(path).get('foreign_key')

        if isinstance(dtype, dict) and 'list' in dtype:
            inner, inner_kwargs = self.type_for(dtype['list'], f'{path}[]', key=False)
            if inner_kwargs:
                self.uses_annotated = True
                inner = f'Annotated[{inner}, Field({", ".join(inner_kwargs)})]'
            ann = f'list[{inner}]'
            if nonnull and p.get('empty') == nonnull:
                kwargs.append('max_length=0')
            elif nonnull and p.get('empty') == 0:
                kwargs.append('min_length=1')
        elif isinstance(dtype, dict) and 'struct' in dtype:
            ann = self.struct(dtype['struct'], path)
        else:
            if dtype not in DTYPES:
                raise ValueError(f'{path}: unsupported dtype {dtype}')
            ann = DTYPES[dtype]
            if dtype == 'Date':
                self.uses_datetime = True
            elif dtype not in {'String', 'Boolean', 'Int64', 'Float64'}:
                self.uses_pl = True
                markers.append(f'PolarsDtype(pl.{dtype})')
                self.imports_base.add('PolarsDtype')
            bioregistry = BIOREGISTRY.search(self.meta_for(path).get('description') or '')
            if bioregistry:
                markers.append(f'Bioregistry({bioregistry.group(1)!r})')
                self.imports_base.add('Bioregistry')
            if ref:
                dataset, _, column = ref.partition('/')
                markers.append(f'Reference({dataset!r}, {column!r})')
                self.imports_base.add('Reference')
            if ann in {'int', 'float', 'datetime.date'} and nonnull:
                lo, hi = p.get('min'), p.get('max')
                if ann == 'datetime.date':
                    lo, hi = (datetime.date.fromisoformat(v) if v else None for v in (lo, hi))
                if lo is not None and lo != float('-inf'):
                    kwargs.append(f'ge={lo!r}')
                if hi is not None and hi != float('inf'):
                    kwargs.append(f'le={hi!r}')
            elif ann == 'str' and nonnull:
                values = p.get('values')
                if values is not None and not key and not ref and nonnull >= ENUM_RATIO * len(values):
                    self.uses_literal = True
                    ann = f'Literal[{", ".join(lit(v) for v in values)}]'
                else:
                    shape = next(
                        (n for n in PATTERNS if n not in SKIP_PATTERNS and p.get(f'pat:{n}') == nonnull), None
                    )
                    if shape:
                        self.patterns.add(shape)
                        kwargs.append(f'pattern={PATTERN_CONSTANTS[shape]}')
                    elif p.get('emptystr') == 0:
                        kwargs.append('min_length=1')
        if markers:
            self.uses_annotated = True
            ann = f'Annotated[{ann}, {", ".join(markers)}]'
        if nullable:
            ann = f'{ann} | None'
        return ann, kwargs

    def struct(self, fields: list, path: str) -> str:
        base = camel(path.replace('[]', '').split('/')[-1])
        name = base
        if name in self.struct_names:
            name = camel(path.replace('[]', '').replace('/', '_'))
        while name in self.struct_names:
            name += 'X'
        self.struct_names.add(name)
        lines = [f'class {name}(StructModel):', f'    """`{path}`."""', '']
        for fname, fdtype in fields:
            lines.append(self.field_line(fname, fdtype, f'{path}/{fname}', top_level=False))
        self.structs.append('\n'.join(lines))
        return name

    def field_line(self, name: str, dtype: Any, path: str, *, top_level: bool) -> str:
        if not name.isidentifier() or keyword.iskeyword(name) or name.startswith('_'):
            raise ValueError(f'{path}: column name {name!r} cannot be a model field')
        meta = self.meta_for(path)
        key = top_level and bool(meta.get('isPrimaryKey')) and self.unique_key
        ann, kwargs = self.type_for(dtype, path, key=key)
        if key:
            self.uses_annotated = True
            self.imports_base.add('PrimaryKey')
            if ann.startswith('Annotated['):
                ann = ann[: -1] + ', PrimaryKey()]'
            else:
                ann = f'Annotated[{ann}, PrimaryKey()]'
        description = BIOREGISTRY.sub('', meta.get('description') or '').strip()
        if description:
            kwargs.append(f'description={lit(description)}')
        if kwargs:
            return f'    {name}: {ann} = Field({", ".join(kwargs)})'
        return f'    {name}: {ann}'

    def render(self, example: dict) -> str:
        fields = [self.field_line(n, d, n, top_level=True) for n, d in self.profile['schema'].items()]
        cls = camel(self.dataset)
        title = self.meta.get('nice_name') or self.dataset.replace('_', ' ').capitalize()
        description = self.meta.get('description') or title
        tags = tuple(self.meta.get('tags') or ())
        self.imports_base |= {'DatasetModel'}
        if self.structs:
            self.imports_base.add('StructModel')
        header = [
            f'"""{title}: the `{self.dataset}` output.',
            '',
            f'Generated from the 26.09 release output ({self.profile["rows"]:,} rows) and its croissant metadata.',
            'Rules are strict to that release: dtypes, nullability, value ranges, allowed values and identifier',
            'shapes are as observed.',
            '"""',
            '',
        ]
        typing = ['ClassVar']
        typing += ['Annotated'] if self.uses_annotated else []
        typing += ['Literal'] if self.uses_literal else []
        imports = (['import datetime'] if self.uses_datetime or 'datetime.date(' in lit(example) else []) + [
            f'from typing import {", ".join(sorted(typing))}', ''
        ]
        if self.uses_pl:
            imports.append('import polars as pl')
        imports.append('from pydantic import Field')
        imports.append('')
        imports.append(f'from pts.schemas.dataset import {", ".join(sorted(self.imports_base))}')
        if self.patterns:
            consts = sorted(PATTERN_CONSTANTS[p] for p in self.patterns)
            imports.append(f'from pts.schemas.outputs.patterns import {", ".join(consts)}')
        body = [
            f'class {cls}(DatasetModel):',
            f'    """{title}."""',
            '',
            f'    dataset_name: ClassVar[str] = {self.dataset!r}',
            f'    dataset_title: ClassVar[str] = {lit(title)}',
            f'    dataset_description: ClassVar[str] = {lit(description)}',
            f'    dataset_tags: ClassVar[tuple[str, ...]] = {tags!r}',
            f'    example: ClassVar[dict] = {lit(example)}',
            '',
            *fields,
        ]
        return '\n'.join(header + imports + ['', ''] + ['\n\n\n'.join(self.structs)] * bool(self.structs)
                         + ['', ''] * bool(self.structs) + body) + '\n'


def composite(dtype: Any, values: list) -> Any:
    """One observed-looking value: the first non-missing primitive; structs and lists built field by field."""
    present = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if isinstance(dtype, dict) and 'struct' in dtype:
        if not present:
            return None
        return {n: composite(d, [v.get(n) for v in present]) for n, d in dtype['struct']}
    if isinstance(dtype, dict) and 'list' in dtype:
        if not present:
            return None
        elements = [e for v in present for e in v]
        if not elements:
            return []
        return [composite(dtype['list'], elements)]
    return present[0] if present else (values[0] if values else None)


def example_row(location: Path, schema: dict) -> dict:
    lf = pl.scan_parquet(location / '**' / '*.parquet', hive_partitioning=False)
    row = {}
    for name, dtype in schema.items():
        col = pl.col(name)
        if isinstance(dtype, dict) and 'list' in dtype:
            sample = lf.select(name).filter(col.list.len() > 0).head(5000).collect(engine='streaming')
        else:
            sample = lf.select(name).filter(col.is_not_null()).head(5000).collect(engine='streaming')
        values = sample[name].to_list() if sample.height else []
        is_list = isinstance(dtype, dict) and 'list' in dtype
        row[name] = composite(dtype, values) if values else ([] if is_list else None)
    return row


def is_unique(location: Path, keys: list[str], rows: int) -> bool:
    lf = pl.scan_parquet(location / '**' / '*.parquet', hive_partitioning=False)
    # a 64-bit hash per key, so 300M+ string keys fit in memory; a collision can only drop a key, never assert one
    return lf.select(pl.struct(keys).hash().n_unique()).collect(engine='streaming').item() == rows


def main() -> None:
    root, profiles, assets, target = (Path(a) for a in sys.argv[1:5])
    distribution = json.loads((assets / 'distribution.json').read_text())
    meta_all = {d['id']: d for d in distribution if isinstance(d, dict) and 'id' in d}
    names = sys.argv[5:] or sorted(p.stem for p in profiles.glob('*.json'))
    for name in names:
        profile = json.loads((profiles / f'{name}.json').read_text())
        recordset = assets / 'recordset' / f'{name}.json'
        croissant = {f['id']: f for f in json.loads(recordset.read_text())} if recordset.exists() else {}
        keys = [k for k, v in croissant.items() if v.get('isPrimaryKey') and '/' not in k and k in profile['schema']]
        null_keys = [k for k in keys if profile['paths'][k]['nulls'] or profile['paths'][k].get('nan')]
        unique = bool(keys) and not null_keys and is_unique(root / name, keys, profile['rows'])
        module = Module(name, profile, croissant, meta_all.get(name, {}), unique)
        try:
            example = example_row(root / name, profile['schema'])
            source = module.render(example)
        except ValueError as e:
            print(f'SKIPPED {name}: {e}')
            continue
        (target / f'{name}.py').write_text(source)
        missing = sorted(k for k in croissant if k.split('/')[0] not in profile['schema'])
        verdict = keys if unique else (('NULLABLE ' if null_keys else 'NOT UNIQUE ') + str(keys) if keys else 'none')
        print(f'{name}: keys {verdict}' + (f'; croissant fields absent from data: {missing}' if missing else ''))


if __name__ == '__main__':
    main()
