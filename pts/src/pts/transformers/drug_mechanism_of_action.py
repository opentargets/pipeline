"""Mechanism of Action processing for drugs.

Prepares the mechanism of action section of the drug object by joining raw
ChEMBL mechanism, target, and reference tables with target/gene information,
and appends the mechanisms inferred from trial texts and the literature.
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.transformers.clinical_mira.staged_outputs import MOA, MOA_LITERATURE
from pts.transformers.drug_mechanism_of_action_candidates import (
    LABEL_COLUMNS,
    SOURCE_AACT,
    SOURCE_LITERATURE,
    build_literature_candidates,
    build_trial_candidates,
    stage_excluded,
)
from pts.transformers.utils import add_parent_chembl_ids
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.target import chembl_accession_lookup

SCHEMA_NAME = 'public'

TABLES = {
    'drug_mechanism': ['mec_id', 'record_id', 'molregno', 'mechanism_of_action', 'tid', 'action_type'],
    # mecref_id is only used to order the input data
    'mechanism_refs': ['mecref_id', 'mec_id', 'ref_type', 'ref_id', 'ref_url'],
    'molecule_dictionary': ['molregno', 'chembl_id'],
    'molecule_hierarchy': ['molregno', 'parent_molregno'],
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type'],
    'target_components': ['tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
}

ORDER_BY = {
    'drug_mechanism': ['mec_id'],
    'mechanism_refs': ['mecref_id'],
    'target_dictionary': ['tid'],
    'target_components': ['tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
}
"""Reads collected into published arrays, which must not come back in plan order."""

CURATION_MANUAL = 'manual'
CURATION_ELECTRONIC = 'electronic'
CLINICAL_TRIALS_SOURCE = 'ClinicalTrials'
CLINICAL_TRIALS_URL = 'https://clinicaltrials.gov/study/'
PUBMED_SOURCE = 'PubMed'
PUBMED_URL = 'http://europepmc.org/abstract/MED/'
PMC_SOURCE = 'PMC'
PMC_URL = 'http://europepmc.org/article/PMC/'
PREPRINT_SOURCE = 'EuropePMC'
PREPRINT_URL = 'https://europepmc.org/article/PPR/'
DOC_KINDS = [
    (CLINICAL_TRIALS_SOURCE, CLINICAL_TRIALS_URL, r'^(NCT.*)$', ''),
    (PUBMED_SOURCE, PUBMED_URL, r'(?i)^pmid(\d+)$', ''),
    (PMC_SOURCE, PMC_URL, r'(?i)^(?:pmid)?pmc(\d+)$', 'PMC'),
    (PREPRINT_SOURCE, PREPRINT_URL, r'(?i)^pmidppr(\d+)$', 'PPR'),
]
"""Each kind of doc id: its reference source and url, the pattern whose group is its id, and the id's prefix."""

def drug_mechanism_of_action(
    source: dict[str, Path],
    destination: Path | dict[str, Path],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Transform ChEMBL mechanism of action data into the Open Targets format.

    Args:
        source: Dictionary with paths to:
            - chembl: Path to the ChEMBL ``pg_dump`` archive.
            - target: Path to the target dataset.
            - aact: Optional path to the staged trial-text mechanism snapshot.
            - literature: Optional path to the staged literature mechanism snapshot.
            - clinical_report, chembl_molecule: Paths to the clinical report and the ChEMBL
              molecule intermediate, read when either snapshot is given.
        destination: Path to write the output parquet file, or a dictionary with:
            - output: Path to write the output parquet file.
            - candidates: Optional path to write the candidates of every source.
            - chembl_report: Optional path to the side file of candidates for ChEMBL.
        settings: Custom settings, read when either snapshot is given:
            - release_min_confidence: Minimum confidence of an inferred mechanism to publish.
            - exclude_stages: For each source, the maximum clinical stages whose inferred
              mechanisms are not published.
            - report_min_confidence, link_share, class_level_genes, suggestion_min_confidence:
              The candidate filters of ``drug_mechanism_of_action_candidates``.
        config: Config object, for ``work_path``.
    """
    logger.info(f'Restoring {list(TABLES)} from {source["chembl"]}')
    # scratch_root: the restore needs gigabytes, and `work_path` is the work disk.
    # See the note in drug_warning.
    tables = read_dump_tables(
        str(source['chembl']), TABLES, schema_name=SCHEMA_NAME, order_by=ORDER_BY, scratch_root=config.work_path
    )

    logger.info(f'Reading target data from {source["target"]}')
    gene_df = scan_dataset(str(source['target'])).select('id', 'proteinIds').collect()

    logger.info('Processing mechanisms of action')
    output_df = process_mechanism_of_action(
        tables['drug_mechanism'],
        tables['mechanism_refs'],
        tables['molecule_dictionary'],
        tables['molecule_hierarchy'],
        tables['target_dictionary'],
        tables['target_components'],
        tables['component_sequences'],
        gene_df,
    )

    destinations = destination if isinstance(destination, dict) else {'output': destination}
    candidates = build_inferred_candidates(source, settings)
    if candidates is not None:
        if destinations.get('candidates') is not None:
            logger.info(f'Writing {candidates.height} candidates to {destinations["candidates"]}')
            write_dataset(candidates, str(destinations['candidates']))
        output_df, report = merge_inferred(
            output_df,
            candidates,
            settings['release_min_confidence'],
            settings['exclude_stages'],
        )
        if destinations.get('chembl_report') is not None:
            logger.info(f'Writing the candidate report for ChEMBL to {destinations["chembl_report"]}')
            write_dataset(report, str(destinations['chembl_report']))

    output_path = destinations['output']

    logger.info(f'Writing mechanism of action to {output_path}')
    write_dataset(output_df, str(output_path))


def process_mechanism_of_action(
    drug_mechanism: pl.DataFrame,
    mechanism_refs: pl.DataFrame,
    molecule_dictionary: pl.DataFrame,
    molecule_hierarchy: pl.DataFrame,
    target_dictionary: pl.DataFrame,
    target_components: pl.DataFrame,
    component_sequences: pl.DataFrame,
    gene_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build mechanisms of action by joining raw ChEMBL tables with target and gene data.

    Args:
        drug_mechanism: Raw ChEMBL drug_mechanism table.
        mechanism_refs: Raw ChEMBL mechanism_refs table.
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        target_dictionary: Raw ChEMBL target_dictionary table.
        target_components: Raw ChEMBL target_components table.
        component_sequences: Raw ChEMBL component_sequences table.
        gene_df: The target dataset's `id` and `proteinIds`.

    Returns:
        Processed mechanism of action DataFrame, one row per surviving `mec_id`.
    """
    ids = add_parent_chembl_ids(drug_mechanism, molecule_dictionary, molecule_hierarchy, key='mec_id')
    mechanism_refs_agg = mechanism_refs.group_by('mec_id', maintain_order=True).agg(
        pl.struct(
            pl.col('ref_type'),
            pl.col('ref_id'),
            pl.col('ref_url'),
        ).alias('mechanism_refs')
    )

    mechanism = (
        _with_target_chembl_id(drug_mechanism, target_dictionary)
        .join(
            molecule_dictionary.select('molregno', pl.col('chembl_id').alias('id')),
            on='molregno',
            how='left',
            maintain_order='left',
        )
        .join(ids, on='mec_id', how='left', maintain_order='left')
        .join(mechanism_refs_agg, on='mec_id', how='left', maintain_order='left')
        .rename({'mechanism_of_action': 'mechanismOfAction', 'action_type': 'actionType'})
        .drop('mec_id', 'molregno')
    )

    references = _chembl_mechanism_references(mechanism)
    target = _chembl_target(target_dictionary, target_components, component_sequences, gene_df)

    result = (
        mechanism
        .join(references, on='id', how='full', coalesce=True, maintain_order='left_right')
        .join(target, on='target_chembl_id', how='full', coalesce=True, maintain_order='left_right')
        .with_columns(pl.col('references').fill_null([]))
        # `parentChemblId` is kept for _consolidate_duplicate_references and dropped there.
        .drop('mechanism_refs', 'record_id', 'target_chembl_id', 'id')
        .filter(
            pl.col('mechanismOfAction').is_not_null()
            & (pl.col('targets').is_not_null() | pl.col('targetName').is_not_null())
            & pl.col('chemblIds').is_not_null()
            & (pl.col('chemblIds').list.len() > 0)
        )
    )

    return _consolidate_duplicate_references(result).with_columns(curation=pl.lit(CURATION_MANUAL))


def build_inferred_candidates(source: dict[str, Path], settings: dict[str, Any]) -> pl.DataFrame | None:
    """Build the candidates of the staged snapshots ``source`` names.

    Args:
        source: The step's sources; ``aact`` and ``literature`` are the optional snapshots.
        settings: The step's settings, with the candidate filters.

    Returns:
        The trial candidates followed by the literature ones, or None if neither snapshot is given.
    """
    present = [name for name in (SOURCE_AACT, SOURCE_LITERATURE) if source.get(name) is not None]
    if not present:
        return None
    clinical_report = (
        scan_dataset(str(source['clinical_report'])).select('id', 'source', 'drugs', 'clinicalStage').collect()
    )
    chembl_molecule = (
        scan_dataset(str(source['chembl_molecule'])).select('id', 'parentId', 'childChemblIds', 'drugType').collect()
    )
    target = scan_dataset(str(source['target'])).select(LABEL_COLUMNS).collect()
    filters = {
        'class_level_genes': settings['class_level_genes'],
        'report_min_confidence': settings['report_min_confidence'],
        'suggestion_min_confidence': settings['suggestion_min_confidence'],
    }
    parts = []
    if SOURCE_AACT in present:
        logger.info(f'Reading the staged trial mechanisms from {source[SOURCE_AACT]}')
        moa = MOA.read(str(source[SOURCE_AACT]))
        parts.append(
            build_trial_candidates(
                moa, clinical_report, chembl_molecule, target, link_share=settings['link_share'], **filters
            )
        )
        logger.info(f'{parts[-1].height} trial candidates of {moa.height} mechanisms')
    if SOURCE_LITERATURE in present:
        logger.info(f'Reading the staged literature mechanisms from {source[SOURCE_LITERATURE]}')
        moa = MOA_LITERATURE.read(str(source[SOURCE_LITERATURE]))
        parts.append(build_literature_candidates(moa, clinical_report, chembl_molecule, target, **filters))
        logger.info(f'{parts[-1].height} literature candidates of {moa.height} mechanisms')
    return pl.concat(parts)


def merge_inferred(
    chembl_rows: pl.DataFrame,
    candidates: pl.DataFrame,
    release_min_confidence: float,
    exclude_stages: dict[str, list[str]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Append the inferred mechanisms to the ChEMBL ones and build the side file for ChEMBL.

    A candidate is published when its drug has no ChEMBL mechanism (``new_moa``), its confidence
    is at least ``release_min_confidence`` and its stage is not excluded for its source. Of the
    published candidates of one drug and gene set, the most confident gives the row, and the
    doc ids of all of them give its references, classified by ``DOC_KINDS``: NCT ids as
    ``ClinicalTrials``, ``pmid<digits>`` as ``PubMed``, ``pmc<digits>`` and ``pmidPMC<digits>`` as
    ``PMC`` and ``pmidPPR<digits>`` as ``EuropePMC``, the prefixes in any case. Other doc ids are
    counted in a warning and not referenced. A row left with no reference is not published. A
    referenced row whose genes are all among those of a higher-ranked referenced row of the same
    drug is not published, see :py:func:`_drop_contained`.

    Args:
        chembl_rows: Output of ``process_mechanism_of_action``.
        candidates: Candidates of every source, with ``CANDIDATE_COLUMNS``.
        release_min_confidence: Minimum confidence of a ``new_moa`` candidate to publish.
        exclude_stages: For each candidate source, the maximum clinical stages whose
            ``new_moa`` candidates are not published.

    Returns:
        The mechanism table with the published candidates appended, and the side file: the most
        confident ``new_moa`` and non-overlapping ``extra_target`` candidate of each source,
        drug and gene set, with ``category`` and ``chemblCurrentTargets``.

    Raises:
        ValueError: If a candidate source has no ``exclude_stages`` entry.
    """
    excluded = stage_excluded(candidates, exclude_stages)
    classified = _classify(chembl_rows, candidates)
    report = _most_confident(classified.filter(pl.col('category').is_not_null()), by_source=True)
    eligible = classified.filter(
        (pl.col('category') == 'new_moa') & (pl.col('confidence') >= release_min_confidence) & ~excluded
    )
    published = (
        _most_confident(eligible, by_source=False).rechunk().with_columns(_references=_references(pl.col('docIds')))
    )
    unknown = published.select(pl.col('docIds').list.eval(~_is_known_doc(pl.element())).list.sum().sum()).item()
    if unknown:
        logger.warning(f'{unknown} doc ids of published candidates are of no known kind and are not referenced')
    unreferenced = published.filter(pl.col('_references').list.len() == 0).height
    if unreferenced:
        logger.warning(f'{unreferenced} candidates not published: they have no reference')
    referenced = published.filter(pl.col('_references').list.len() > 0)
    kept = _drop_contained(referenced)
    if kept.height < referenced.height:
        logger.info(
            f'{referenced.height - kept.height} inferred rows not published: '
            'their genes are a subset of a higher-ranked row of the same drug'
        )
    appended = kept.select(
        'chemblIds',
        'mechanismOfAction',
        'actionType',
        pl.col('targetFromSource').alias('targetName'),
        'targetType',
        'targets',
        pl.col('_references').alias('references'),
        pl.lit(CURATION_ELECTRONIC).alias('curation'),
    ).select([pl.col(name).cast(dtype) for name, dtype in chembl_rows.schema.items()])
    return pl.concat([chembl_rows, appended]), report


def _drop_contained(published: pl.DataFrame) -> pl.DataFrame:
    """Keep, of each drug's rows, those whose genes are not a subset of a row ranked above them.

    A drug's rows are ranked by confidence, then by number of doc ids, both descending, then by
    their sorted genes joined as a string. A row is dropped when its genes are a subset of those
    of any row ranked before it; rows that only partly overlap are kept. The kept rows stay in
    their input order.
    """
    ranked = (
        published
        .with_row_index('_row')
        .with_columns(
            _n_docs=pl.col('docIds').list.len(),
            _targets_key=pl.col('targets').fill_null([]).list.sort().list.join(','),
        )
        .sort('parentChemblId', 'confidence', '_n_docs', '_targets_key', descending=[False, True, True, False])
    )
    above: dict[str, list[set[str]]] = {}
    kept = []
    for row, parent, targets in ranked.select('_row', 'parentChemblId', 'targets').iter_rows():
        genes = set(targets or [])
        ranked_above = above.setdefault(parent, [])
        if not any(genes <= other for other in ranked_above):
            kept.append(row)
        ranked_above.append(genes)
    return published.with_row_index('_row').filter(pl.col('_row').is_in(kept)).drop('_row')


def _classify(chembl_rows: pl.DataFrame, candidates: pl.DataFrame) -> pl.DataFrame:
    """Add ``chemblCurrentTargets`` and ``category`` to each candidate.

    ``category`` is ``new_moa`` when no molecule of the candidate's drug has a ChEMBL mechanism,
    ``extra_target`` when they have one but no gene of the candidate is among their targets, and
    null otherwise.
    """
    indexed = candidates.with_row_index('_row')
    chembl_targets = (
        chembl_rows
        .select('chemblIds', 'targets')
        .explode('chemblIds')
        .group_by('chemblIds')
        .agg(pl.col('targets').explode().drop_nulls().unique().alias('chemblTargets'))
        .rename({'chemblIds': '_drug_id'})
    )
    current = (
        indexed
        .select(
            '_row',
            _drug_id=pl.concat_list(pl.col('parentChemblId').cast(pl.String), pl.col('chemblIds').fill_null([])),
        )
        .explode('_drug_id')
        .unique()
        .join(chembl_targets, on='_drug_id', how='inner')
        .group_by('_row')
        .agg(pl.col('chemblTargets').explode().drop_nulls().unique().sort().alias('chemblCurrentTargets'))
    )
    return (
        indexed
        .join(current, on='_row', how='left', maintain_order='left')
        .with_columns(_has_mechanism=pl.col('chemblCurrentTargets').is_not_null())
        .with_columns(chemblCurrentTargets=pl.col('chemblCurrentTargets').fill_null([]))
        .with_columns(
            category=pl
            .when(~pl.col('_has_mechanism'))
            .then(pl.lit('new_moa'))
            .when(pl.col('targets').list.set_intersection(pl.col('chemblCurrentTargets')).list.len() == 0)
            .then(pl.lit('extra_target'))
        )
        .drop('_has_mechanism', '_row')
    )


def _most_confident(candidates: pl.DataFrame, *, by_source: bool) -> pl.DataFrame:
    """Keep the most confident candidate of each drug and gene set, and of each source if ``by_source``.

    Across sources, ``docIds`` becomes the sorted union of the group's doc ids.
    """
    keys = ['source', 'parentChemblId', '_targets_key'] if by_source else ['parentChemblId', '_targets_key']
    doc_ids = pl.col('docIds').first() if by_source else pl.col('docIds').list.explode().unique().sort()
    return (
        candidates
        .with_columns(_targets_key=pl.col('targets').list.sort())
        .sort('confidence', descending=True, maintain_order=True)
        .group_by(keys, maintain_order=True)
        .agg(pl.exclude('docIds').first(), doc_ids.alias('docIds'))
        .select(candidates.columns)
    )


def _is_known_doc(doc_id: pl.Expr) -> pl.Expr:
    """Whether a doc id is of one of the ``DOC_KINDS``."""
    return pl.any_horizontal(doc_id.str.contains(pattern) for _, _, pattern, _ in DOC_KINDS)


def _references(doc_ids: pl.Expr) -> pl.Expr:
    """One reference per kind of doc id present, in ``DOC_KINDS`` order.

    PubMed ids sort by length, then as strings, which orders them as numbers; the others sort as strings.
    """

    def reference(source: str, url: str, pattern: str, prefix: str) -> pl.Expr:
        ids = (
            doc_ids.list
            .eval(pl.element().str.extract(pattern, 1).drop_nulls())
            .list.eval(pl.lit(prefix) + pl.element())
            .list.unique()
            .list.sort()
        )
        if source == PUBMED_SOURCE:
            ids = ids.list.eval(pl.element().sort_by(pl.element().str.len_chars(), maintain_order=True))
        return pl.struct(
            pl.lit(source).alias('source'),
            ids.alias('ids'),
            ids.list.eval(pl.lit(url) + pl.element()).alias('urls'),
        )

    return pl.concat_list(reference(*kind) for kind in DOC_KINDS).list.eval(
        pl.element().filter(pl.element().struct.field('ids').list.len() > 0)
    )


def _with_target_chembl_id(mechanism: pl.DataFrame, target_dictionary: pl.DataFrame) -> pl.DataFrame:
    """Resolve each mechanism's `tid` to a `target_chembl_id`.

    Uses a left join, so a null or unmatched `tid` yields a null `target_chembl_id`
    rather than dropping the mechanism row.

    Args:
        mechanism: Raw ChEMBL drug_mechanism table.
        target_dictionary: Raw ChEMBL target_dictionary table.

    Returns:
        `mechanism` with `tid` replaced by `target_chembl_id`.
    """
    return mechanism.join(
        target_dictionary.select('tid', pl.col('chembl_id').alias('target_chembl_id')),
        on='tid',
        how='left',
        maintain_order='left',
    ).drop('tid')


def _chembl_mechanism_references(df: pl.DataFrame) -> pl.DataFrame:
    """Extract and structure references from mechanism data.

    Args:
        df: Mechanism DataFrame with id and mechanism_refs columns.

    Returns:
        DataFrame with id and references columns. Only ids that have at least one
        reference are present -- the caller is responsible for filling the gap
        left by ids with none.
    """
    return (
        df
        .select('id', 'mechanism_refs')
        .explode('mechanism_refs', empty_as_null=True)
        .filter(pl.col('mechanism_refs').is_not_null())
        .unnest('mechanism_refs')
        .group_by('id', 'ref_type', maintain_order=True)
        .agg(
            # drop_nulls is load-bearing: a reference with no id or url (an 'Expert' or
            # 'KEGG' entry, say) would otherwise contribute a `None` element, and since
            # the two columns are null in different rows, `ids` and `urls` would come out
            # positionally desynchronised inside the published struct.
            pl.col('ref_id').drop_nulls().alias('ids'),
            pl.col('ref_url').drop_nulls().alias('urls'),
        )
        .with_columns(
            references=pl.struct(
                pl.col('ref_type').alias('source'),
                pl.col('ids'),
                pl.col('urls'),
            )
        )
        .group_by('id', maintain_order=True)
        .agg(pl.col('references'))
    )


def _chembl_target(
    target_dictionary: pl.DataFrame,
    target_components: pl.DataFrame,
    component_sequences: pl.DataFrame,
    gene_df: pl.DataFrame,
) -> pl.DataFrame:
    """Process ChEMBL target data and join with gene information.

    Args:
        target_dictionary: Raw ChEMBL target_dictionary table.
        target_components: Raw ChEMBL target_components table.
        component_sequences: Raw ChEMBL component_sequences table.
        gene_df: The target dataset's `id` and `proteinIds`.

    Returns:
        DataFrame with target_chembl_id, targetName, targetType, and targets.
    """
    target_components_flat = (
        target_components
        .join(component_sequences, on='component_id', how='inner', maintain_order='left')
        .join(target_dictionary, on='tid', how='inner', maintain_order='left')
        .filter(pl.col('accession').is_not_null())
        .select(
            pl.col('pref_name').alias('targetName'),
            pl.col('accession').alias('uniprot_id'),
            pl.col('target_type').str.to_lowercase().alias('targetType'),
            pl.col('chembl_id').alias('target_chembl_id'),
        )
        .with_row_index('component_order')
    )

    gene_lookup = chembl_accession_lookup(gene_df)

    # genes sharing an accession come in id order
    joined = target_components_flat.join(gene_lookup, on='uniprot_id', how='left').sort('component_order', 'geneId')

    return joined.group_by('target_chembl_id', 'targetName', 'targetType', maintain_order=True).agg(
        pl.col('geneId').drop_nulls().unique(maintain_order=True).alias('targets')
    )


def _consolidate_duplicate_references(df: pl.DataFrame) -> pl.DataFrame:
    """Consolidate mechanism rows that are identical **for the same drug**.

    ChEMBL propagates the mechanism for their child/salt molecules in
    ``_metadata.all_molecule_chembl_ids`` to include the parent. When the parent has
    the same mechanism, these rows carry identical display information and differ only
    by ``chemblIds`` (and the per-molecule ``id``, already dropped). This step avoids the
    duplication on the mechanism for the parent once the data is exploded
    by ``chemblId`` downstream.

    ``parentChemblId`` is in the grouping key, and is what confines the merge to one
    drug. The rest of the key is display information -- mechanism, action type, target --
    which every drug in a pharmacological class shares, so without the anchor a whole
    class merges into one row and each of its drugs publishes the class's references
    rather than its own. The anchor is the *parent*, not the molecule, so that sibling
    salts of one drug still merge, which is what the step exists to do. It is internal,
    and dropped on the way out.

    ``chemblIds`` and ``references`` are deduplicated at different granularities, which is
    deliberate: ``chemblIds`` is flattened and then distinct-ed element by element, while
    ``references`` dedupes whole per-row lists first and only then concatenates the
    survivors. Two rows whose reference lists are not byte-identical can therefore still
    contribute overlapping reference structs. That asymmetry is present in the published
    data, so it is preserved rather than tidied up.
    """
    key_cols = [c for c in df.columns if c not in ('references', 'chemblIds')]
    output_cols = [c for c in df.columns if c != 'parentChemblId']
    return (
        df
        .group_by(key_cols, maintain_order=True)
        .agg(
            pl
            .col('chemblIds')
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique(maintain_order=True)
            .alias('chemblIds'),
            pl
            .col('references')
            .unique(maintain_order=True)
            .list.explode(keep_nulls=False, empty_as_null=False)
            .alias('references'),
        )
        .select(output_cols)
    )
