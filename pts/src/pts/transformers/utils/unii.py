"""FDA UNII annotation of ChEMBL molecules.

The FDA UNII lists (``UNII_Data.zip``, ``UNIIs.zip``) name every substance in the FDA
Substance Registration System. :mod:`pts.transformers.chembl_molecule` uses the substances
that carry an INN in two ways:

- :func:`annotate_molecules` matches them to molecules by standard InChIKey and adds the
  INN as the name of an unnamed molecule, the official names and codes as synonyms, the
  brand names as trade names, and the UNII and NCIt ids as cross-references. A label
  another molecule family already carries is not added.
- :func:`drop_foreign_inn_synonyms` removes a trial (AACT) synonym that is the INN of a
  substance another molecule family already carries.
"""

import zipfile
from io import BytesIO

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.unii import unii_annotation_schema, unii_inn_schema
from pts.transformers.utils.aact_synonyms import AACT_SOURCE

UNII_SOURCE = 'UNII'
NCIT_SOURCE = 'NCIt'

RECORDS_MEMBER = 'UNII_Records_'
NAMES_MEMBER = 'UNII_Names_'

MIN_LABEL_LENGTH = 4
"""Shorter official names and codes are left out: too ambiguous to map trial text."""

OFFICIAL, CODE, BRAND = 'of', 'cd', 'bn'

_TAGS = r'(?:\s*\[[^\]]*\])+$'
_INN_TAG = r'\[INN\]$'


def read_unii(path: str, member_prefix: str) -> pl.DataFrame:
    """Read the tab-separated list in a UNII zip, every column as a string.

    Args:
        path: The zip, local or remote.
        member_prefix: Start of the list's file name in the zip (`RECORDS_MEMBER`, `NAMES_MEMBER`).

    Returns:
        The list as a DataFrame.

    Raises:
        ValueError: If the zip has no such list, or more than one.
    """
    content, _ = StorageHandle(path).read()
    with zipfile.ZipFile(BytesIO(content)) as archive:
        members = [n for n in archive.namelist() if n.startswith(member_prefix) and n.endswith('.txt')]
        if len(members) != 1:
            raise ValueError(f'expected one {member_prefix}*.txt in {path}, found {members}')
        data = archive.read(members[0])
    return pl.read_csv(data, separator='\t', infer_schema=False, quote_char=None)


def _strip_tags(name: pl.Expr) -> pl.Expr:
    """The name without its trailing origin tags (`[INN]`, `[USAN]`, `[WHO-DD]`)."""
    return name.str.replace(_TAGS, '').str.strip_chars()


def _norm(label: pl.Expr) -> pl.Expr:
    """Lowercased, whitespace collapsed: the form AACT synonyms are stored in."""
    return label.str.strip_chars().str.replace_all(r'\s+', ' ').str.to_lowercase()


def inn_substances(records: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """The UNII substances with an INN number, one row per INN name.

    Args:
        records: The UNII records list.
        names: The UNII names list.

    Returns:
        DataFrame of :data:`pts.schemas.unii.unii_inn_schema`.
    """
    inn_records = records.filter(pl.col('INN_ID').is_not_null()).select('UNII', 'INCHIKEY')
    return (
        names
        .filter(pl.col('NAME').str.contains(_INN_TAG))
        .join(inn_records, on='UNII', how='inner')
        .select(
            pl.col('UNII').alias('unii'),
            _strip_tags(pl.col('NAME')).alias('inn'),
            pl.col('INCHIKEY').alias('inchiKey'),
        )
        .filter(pl.col('inn').str.len_chars() > 0)
        .unique(maintain_order=True)
        .cast(unii_inn_schema)
    )


def unii_annotations(records: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """One row per InChIKey of an INN substance: its INN, names, codes, brands and ids.

    An InChIKey shared by two INN substances is left out: it cannot say which one a
    molecule is. A substance with two INN names takes the first in alphabetical order.

    Args:
        records: The UNII records list.
        names: The UNII names list.

    Returns:
        DataFrame of :data:`pts.schemas.unii.unii_annotation_schema`; `synonyms` lists
        the INN first, then official names and codes, deduplicated case-insensitively.
    """
    inns = inn_substances(records, names).drop_nulls('inchiKey')
    unique_keys = inns.group_by('inchiKey').agg(pl.col('unii').n_unique().alias('n')).filter(pl.col('n') == 1)
    inns = (
        inns
        .join(unique_keys.select('inchiKey'), on='inchiKey', how='semi')
        .sort('unii', 'inn')
        .unique('unii', keep='first', maintain_order=True)
    )

    labels = (
        names
        .join(inns.select('unii'), left_on='UNII', right_on='unii', how='semi')
        .filter(pl.col('TYPE').is_in([OFFICIAL, CODE, BRAND]))
        .select(pl.col('UNII').alias('unii'), pl.col('TYPE').alias('type'), _strip_tags(pl.col('NAME')).alias('label'))
        .filter(pl.col('label').str.len_chars() >= MIN_LABEL_LENGTH)
    )
    # the INN leads the synonyms, then official names, then codes
    ranked = pl.concat([
        inns.select('unii', type=pl.lit(OFFICIAL), label='inn', rank=pl.lit(0)),
        labels.filter(pl.col('type') != BRAND).with_columns(
            rank=pl.when(pl.col('type') == OFFICIAL).then(1).otherwise(2)
        ),
    ])
    synonyms = (
        ranked
        .sort('unii', 'rank', 'label')
        .unique(['unii', _norm(pl.col('label')).alias('key')], keep='first', maintain_order=True)
        .group_by('unii', maintain_order=True)
        .agg(pl.col('label').alias('synonyms'))
    )
    brands = (
        labels
        .filter(pl.col('type') == BRAND)
        .sort('unii', 'label')
        .unique(['unii', _norm(pl.col('label')).alias('key')], keep='first', maintain_order=True)
        .group_by('unii', maintain_order=True)
        .agg(pl.col('label').alias('tradeNames'))
    )
    ncit = records.select(pl.col('UNII').alias('unii'), pl.col('NCIT').alias('ncit'))
    return (
        inns
        .join(synonyms, on='unii', how='left')
        .join(brands, on='unii', how='left')
        .join(ncit, on='unii', how='left')
        .select(
            'inchiKey',
            'unii',
            'inn',
            pl.col('synonyms').fill_null([]),
            pl.col('tradeNames').fill_null([]),
            'ncit',
        )
        .cast(unii_annotation_schema)
    )


def _labels(molecules: pl.DataFrame, column: str) -> pl.DataFrame:
    """One row per (`id`, label struct `s`) of a list column; `key` is the normalized label."""
    return (
        molecules
        .select('id', s=column)
        .explode('s')
        .drop_nulls('s')
        .with_columns(key=_norm(pl.col('s').struct.field('label')))
    )


def _with_labels(molecules: pl.DataFrame, column: str, rows: pl.DataFrame) -> pl.DataFrame:
    """`molecules` with `column` rebuilt from `rows` (`id`, `s`), sorted; empty for an id without rows."""
    grouped = rows.group_by('id').agg(pl.col('s').alias('_new'))
    return (
        molecules
        .join(grouped, on='id', how='left', validate='m:1')
        .with_columns(pl.col('_new').fill_null([]).list.sort().alias(column))
        .drop('_new')
    )


def _blocked(molecules: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """The (`id`, `key`) candidates that would give a label to a second molecule family.

    A candidate is blocked when a molecule of another family already has the label (name,
    synonym or trade name) or when another candidate gives it to another family.
    """
    families = molecules.select('id', family=pl.coalesce('parentId', 'id')).unique('id')
    carried = pl.concat([
        molecules.select('id', key=_norm(pl.col('name'))).drop_nulls('key'),
        _labels(molecules, 'synonyms').select('id', 'key'),
        _labels(molecules, 'tradeNames').select('id', 'key'),
    ]).join(candidates.select('key').unique(), on='key', how='semi')
    holders = pl.concat([carried, candidates.select('id', 'key')]).join(families, on='id').select('key', 'family')
    shared = holders.unique().group_by('key').agg(pl.len().alias('n')).filter(pl.col('n') > 1).select('key')
    return candidates.join(shared, on='key', how='semi').select('id', 'key').unique()


def annotate_molecules(molecules: pl.DataFrame, annotations: pl.DataFrame) -> pl.DataFrame:
    """Add the UNII annotations of a molecule's InChIKey to it.

    A molecule without a ChEMBL name takes the INN, uppercased as ChEMBL names are.
    UNII labels a molecule already has (any source, case-insensitive) are not repeated;
    the labels it has are kept as they are. The INN, a synonym or a trade name is not
    added when that would put the label on two molecule families (the molecule, its
    parent and the parent's other children): another family already carries it, or two
    families match the substance; the merge of such duplicates is left to a later step.
    The UNII and NCIt cross-references come from the structure match alone. A matched
    molecule's null `synonyms` or `tradeNames` comes back empty; the other molecules
    come back unchanged, after the matched ones.

    Args:
        molecules: Molecules with id, inchiKey, name, parentId, synonyms, tradeNames and
            crossReferences columns; `name` is null when ChEMBL has none.
        annotations: As returned by :func:`unii_annotations`.

    Returns:
        `molecules` with the same columns.
    """
    matched = (
        molecules.select('id', 'inchiKey').unique('id').join(annotations, on='inchiKey', how='inner', validate='m:1')
    )
    candidates = (
        pl
        .concat([
            matched.select('id', column=pl.lit('name'), label=pl.col('inn').str.to_uppercase()),
            *(
                matched.select('id', column=pl.lit(column), label=column).explode('label').drop_nulls('label')
                for column in ('synonyms', 'tradeNames')
            ),
        ])
        .with_columns(key=_norm(pl.col('label')))
        .unique(['id', 'column', 'key'], keep='first', maintain_order=True)
    )
    candidates = candidates.join(_blocked(molecules, candidates), on=['id', 'key'], how='anti')

    columns = molecules.columns
    # only the matched molecules' lists are rebuilt
    is_matched = pl.col('id').is_in(matched['id'].implode())
    out = molecules.filter(is_matched)
    for column in ('synonyms', 'tradeNames'):
        existing = _labels(out, column)
        new = (
            candidates
            .filter(pl.col('column') == column)
            .join(existing.select('id', 'key'), on=['id', 'key'], how='anti')
            .select('id', s=pl.struct('label', pl.lit(UNII_SOURCE).alias('source')))
        )
        out = _with_labels(out, column, pl.concat([existing.select('id', 's'), new]))
    inn = candidates.filter(pl.col('column') == 'name').select('id', _inn='label').unique('id')
    xref = matched.select(
        'id',
        pl
        .concat_list(
            pl.struct(pl.lit(UNII_SOURCE).alias('source'), pl.concat_list('unii').alias('ids')),
            pl
            .when(pl.col('ncit').is_not_null())
            .then(pl.struct(pl.lit(NCIT_SOURCE).alias('source'), pl.concat_list('ncit').alias('ids')))
            .otherwise(None),
        )
        .list.drop_nulls()
        .alias('_xref'),
    ).unique('id', maintain_order=True)
    out = (
        out
        .join(inn, on='id', how='left', validate='m:1')
        .join(xref, on='id', how='left', validate='m:1')
        .with_columns(
            name=pl.coalesce('name', '_inn'),
            crossReferences=pl
            .when(pl.col('_xref').is_null())
            .then('crossReferences')
            .otherwise(pl.concat_list(pl.col('crossReferences').fill_null([]), '_xref')),
        )
        .select(columns)
    )
    return pl.concat([molecules.filter(~is_matched), out])


def drop_foreign_inn_synonyms(molecules: pl.DataFrame, inns: pl.DataFrame) -> pl.DataFrame:
    """Drop the AACT synonyms that are another molecule family's INN.

    An INN belongs to the families whose molecules carry it as a name, a ChEMBL or UNII
    synonym or a trade name, or whose structure is the INN substance's. An AACT synonym
    equal to an INN goes when the INN belongs to some family and not to the molecule's
    own (the molecule, its parent and the parent's other children). An INN no family
    carries stays: the trial may be what links a code-named molecule to its INN.

    Args:
        molecules: Molecules with id, parentId, inchiKey, name, synonyms and tradeNames.
        inns: As returned by :func:`inn_substances`.

    Returns:
        `molecules` with the same columns, fewer AACT synonyms.
    """
    family = pl.coalesce('parentId', 'id').alias('family')
    inn_keys = inns.select(_norm(pl.col('inn')).alias('key'), 'inchiKey').unique()

    families = molecules.select('id', family).unique('id')
    synonyms = _labels(molecules, 'synonyms')
    curated = (
        pl
        .concat([
            molecules.select('id', key=_norm(pl.col('name'))),
            synonyms.filter(pl.col('s').struct.field('source') != AACT_SOURCE).select('id', 'key'),
            _labels(molecules, 'tradeNames').select('id', 'key'),
        ])
        .join(families, on='id', how='left')
        .select('family', 'key')
    )
    by_structure = (
        molecules
        .select(family, 'inchiKey')
        .drop_nulls('inchiKey')
        .join(inn_keys.drop_nulls('inchiKey'), on='inchiKey')
        .select('family', 'key')
    )
    owners = pl.concat([curated.join(inn_keys.select('key').unique(), on='key', how='semi'), by_structure]).unique()

    aact = synonyms.filter(pl.col('s').struct.field('source') == AACT_SOURCE).join(families, on='id', how='left')
    owned = aact.join(owners.select('key').unique(), on='key', how='semi')
    foreign = owned.join(owners, on=['family', 'key'], how='anti').select('id', 'key').unique()
    if foreign.is_empty():
        return molecules
    kept = synonyms.join(
        foreign.with_columns(source=pl.lit(AACT_SOURCE)),
        left_on=['id', 'key', pl.col('s').struct.field('source')],
        right_on=['id', 'key', 'source'],
        how='anti',
    ).select('id', 's')
    return _with_labels(molecules, 'synonyms', kept)
