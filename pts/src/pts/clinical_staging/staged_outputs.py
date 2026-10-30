"""The kinds of model output that staging computes and the release reads.

Each kind names the Hugging Face models it runs, the tuned constants of the code
around them (``params``), and the columns of its snapshot. Any other change to a
kind's compute needs a ``logic_version`` bump. A kind's tag identifies one output
revision::

    <hub revision>+<hub revision>-p<digest>

The hub revisions come in the kind's model order; the digest is the first 8 hex of
the sha256 of the parameters and ``logic_version``. For these kinds the tag is both
the record store's ``model_revision`` and the directory a snapshot is published under.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import polars as pl
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel

from pts.record_store import LIST_OUTPUT, map_texts, publish_snapshot, tracked
from pts.transformers.clinical.ner import (
    BIOBERT_LABELS,
    CHEMICAL_MODEL,
    DISEASE_DROP_DIGITS,
    DISEASE_LABELS,
    DISEASE_MIN_LENGTH,
    DISEASE_MODEL,
    DISEASE_NER_MODEL,
    DRUG_NAME_PATTERNS,
    DRUG_NER_MODEL,
    DRUGTEMIST_LABELS,
    DRUGTEMIST_MODEL,
)
from pts.transformers.clinical.providers.aact_llm import TRIAL_FIELDS
from pts.transformers.clinical.schemas import ClinicalReportExtractionSchema
from pts.transformers.clinical.stop_reasons import STOP_REASONS_MODEL, STOP_REASONS_THRESHOLD
from pts.transformers.utils.dataset import scan_dataset

if TYPE_CHECKING:
    from collections.abc import Callable

    from otter.config.model import Config

METADATA_KEYS = ('kind', 'model', 'tag', 'model_revisions')
"""The keys every snapshot's metadata holds."""

MAX_EXAMPLES = 5
"""Keys quoted in the error raised for conflicting snapshots."""


def schema_digest(model_cls: type[BaseModel]) -> str:
    """The full sha256 hex of the sorted JSON schema, by alias, of ``model_cls``."""
    return hashlib.sha256(
        json.dumps(model_cls.model_json_schema(by_alias=True), sort_keys=True).encode('utf-8')
    ).hexdigest()


def read_metadata(source: str, uri: str) -> dict[str, Any]:
    """The metadata of the snapshot ``uri``, named ``source`` in errors.

    Raises:
        ValueError: if the metadata lacks a key of ``METADATA_KEYS``.
    """
    metadata = json.loads(StorageHandle(f'{uri}/metadata.json').read_text()[0])
    missing = [key for key in METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f'snapshot {source} metadata lacks {missing}')
    return metadata


@dataclass
class StagedResult:
    """What one staging of a kind produced.

    ``frame`` has the snapshot columns; ``computed`` says whether any text was sent to
    the model, ``published`` whether a new snapshot was written.
    """

    frame: pl.DataFrame
    computed: bool
    published: bool
    tag: str

    def stats(self) -> dict[str, Any]:
        """Return the number of texts, whether anything was computed or published, and the tag."""
        return {'texts': self.frame.height, 'computed': self.computed, 'published': self.published, 'tag': self.tag}


@dataclass(frozen=True)
class StagedOutput:
    """One kind of staged model output.

    ``name`` is also the record store's name and the destination key. ``columns``
    maps a record column to its snapshot column; ``key`` is the snapshot column
    that identifies a row.
    """

    name: str
    model: str
    hub_models: tuple[str, ...]
    params: dict[str, Any]
    logic_version: int
    columns: dict[str, str]
    key: str

    @property
    def digest(self) -> str:
        """The first 8 hex of the sha256 of ``params`` and ``logic_version``."""
        document = json.dumps({'logic_version': self.logic_version, 'params': self.params}, sort_keys=True)
        return hashlib.sha256(document.encode('utf-8')).hexdigest()[:8]

    def require(self, revisions: dict[str, str]) -> None:
        """Check that ``revisions`` names every hub model of this kind.

        Raises:
            ValueError: listing the missing models.
        """
        missing = [m for m in self.hub_models if m not in revisions]
        if missing:
            raise ValueError(f'{self.name} needs models that are not staged: {missing}')

    def tag(self, revisions: dict[str, str]) -> str:
        """The output revision of this kind at the hub ``revisions``."""
        self.require(revisions)
        return '+'.join(revisions[m] for m in self.hub_models) + f'-p{self.digest}'

    def stage(
        self,
        texts: pl.Series,
        compute: Callable[[list[str]], list[tuple[Any, str | None]]],
        revisions: dict[str, str],
        store_root: str,
        destination: str,
        run_id: str,
        config: Config,
        max_attempts: int,
        extra_metadata: dict[str, str] | None = None,
    ) -> StagedResult:
        """Map ``texts`` through the store ``{store_root}/{name}`` and publish the snapshot ``{destination}/{tag}``.

        Args:
            texts: The texts to map.
            compute: Maps a list of texts to one ``(output, None)`` or ``(None, error)`` per text.
            revisions: The hub revision of each staged model.
            store_root: Directory holding one record store per kind.
            destination: Directory holding one snapshot per tag.
            run_id: Identifies the store run, if anything is computed.
            config: The otter config, used to resolve storage.
            max_attempts: Failed records after which a text is no longer retried.
            extra_metadata: Further keys for the snapshot's metadata.

        Returns:
            StagedResult: The snapshot contents, its tag, and whether anything was computed or published.
        """
        tag = self.tag(revisions)
        calls: list[int] = []
        records = map_texts(
            texts,
            tracked(compute, calls),
            f'{store_root}/{self.name}',
            self.model,
            tag,
            run_id,
            config,
            max_attempts,
        )
        frame = records.select(pl.col(record).alias(column) for record, column in self.columns.items())
        metadata = {
            'kind': self.name,
            'model': self.model,
            'tag': tag,
            'model_revisions': {m: revisions[m] for m in self.hub_models},
            **(extra_metadata or {}),
        }
        published = publish_snapshot(frame, f'{destination}/{tag}', metadata, config)
        return StagedResult(frame=frame, computed=bool(calls), published=published, tag=tag)

    def read(self, snapshots: dict[str, str], expect: dict[str, dict[str, str]] | None = None) -> pl.DataFrame:
        """The snapshot columns of ``snapshots``, merged into one row per ``key``.

        A single snapshot is returned as staged. Two or more are concatenated, their list
        outputs sorted, and identical rows dropped.

        Args:
            snapshots: Each snapshot directory, keyed by its source name.
            expect: Metadata values, by source name, that the source's snapshot must hold. A source
                without an entry is not checked.

        Returns:
            The snapshot columns.

        Raises:
            ValueError: if a snapshot's metadata lacks a key or differs from ``expect``, is of
                another kind or was staged by code with other parameters, if the snapshots
                have different tags, or if a key has different results in two snapshots.
        """
        tags = {}
        for source, uri in sorted(snapshots.items()):
            metadata = read_metadata(source, uri)
            for key, value in (expect or {}).get(source, {}).items():
                if metadata.get(key) != value:
                    raise ValueError(f'snapshot {source} {key} expected {value!r}, found {metadata.get(key)!r}')
            if metadata['kind'] != self.name:
                raise ValueError(f'snapshot {source} is of kind {metadata["kind"]!r}, not {self.name!r}')
            digest = metadata['tag'].rpartition('-p')[2]
            if digest != self.digest:
                raise ValueError(f'snapshot {source} was staged with digest {digest!r}, this code has {self.digest!r}')
            tags[source] = metadata['tag']
        if len(set(tags.values())) > 1:
            raise ValueError(f'{self.name} snapshots have different tags: {tags}')

        names = list(self.columns.values())
        frames = [scan_dataset(snapshots[source]).select(names).collect() for source in sorted(snapshots)]
        if len(frames) == 1:
            return frames[0]

        output = self.columns['output']
        schema = pl.Schema({name: LIST_OUTPUT if name == output else pl.String for name in names})
        merged = pl.concat([frame.cast(schema) for frame in frames]).with_columns(pl.col(output).list.sort()).unique()
        conflicts = merged.filter(pl.col(self.key).is_duplicated())[self.key].unique().sort()
        if conflicts.len() > 0:
            raise ValueError(
                f'{conflicts.len()} {self.name} keys staged with different results, '
                f'e.g. {conflicts.head(MAX_EXAMPLES).to_list()}'
            )
        return merged.sort(self.key)


@dataclass(frozen=True)
class ExtractionOutput:
    """The LLM extraction of the AACT trials, tagged by its response schema and the code around it.

    Unlike a ``StagedOutput`` it runs no Hugging Face model. Its tag is the first 12 hex of the
    response schema digest, then ``-p`` and the first 8 hex of the sha256 of ``logic_version``,
    ``params['trial_fields']`` and ``sample_size``. Its store revision is the full schema
    digest, not its tag.
    """

    name: str
    model: str
    params: dict[str, Any]
    logic_version: int

    def tag(self, digest: str, sample_size: int | None) -> str:
        """The output revision of an extraction against schema ``digest`` of ``sample_size`` trials, or all if None."""
        document = json.dumps(
            {
                'logic_version': self.logic_version,
                'trial_fields': self.params['trial_fields'],
                'sample_size': sample_size,
            },
            sort_keys=True,
        )
        return f'{digest[:12]}-p{hashlib.sha256(document.encode("utf-8")).hexdigest()[:8]}'

    def release_tag(self) -> str:
        """The tag of a full extraction by this code's response schema, the only one a release reads."""
        return self.tag(schema_digest(ClinicalReportExtractionSchema), None)

    def metadata(self, digest: str, sample_size: int | None) -> dict[str, Any]:
        """The snapshot metadata of an extraction against schema ``digest`` of ``sample_size`` trials."""
        return {
            'kind': self.name,
            'model': self.model,
            'tag': self.tag(digest, sample_size),
            'model_revisions': {self.model: digest},
        }

    def read(self, uri: str) -> pl.DataFrame:
        """The extraction snapshot ``uri``.

        Raises:
            ValueError: if its metadata lacks a key, is of another kind, or its tag is not ``release_tag()``.
        """
        metadata = read_metadata(self.name, uri)
        if metadata['kind'] != self.name:
            raise ValueError(f'snapshot {self.name} is of kind {metadata["kind"]!r}, not {self.name!r}')
        if metadata['tag'] != self.release_tag():
            raise ValueError(
                f'snapshot {self.name} has tag {metadata["tag"]!r}, this code expects {self.release_tag()!r}'
            )
        return scan_dataset(uri).collect()


STOP_REASONS = StagedOutput(
    name='stop_reasons',
    model=STOP_REASONS_MODEL,
    hub_models=(STOP_REASONS_MODEL,),
    params={'threshold': STOP_REASONS_THRESHOLD},
    logic_version=1,
    columns={'text': 'text', 'text_sha256': 'text_sha256', 'output': 'categories'},
    key='text',
)

DRUG_ENTITIES = StagedOutput(
    name='drug_entities',
    model=DRUG_NER_MODEL,
    hub_models=(CHEMICAL_MODEL, DRUGTEMIST_MODEL),
    params={
        'patterns': DRUG_NAME_PATTERNS,
        'biobert_labels': BIOBERT_LABELS,
        'drugtemist_labels': DRUGTEMIST_LABELS,
    },
    logic_version=1,
    columns={'text': 'label', 'output': 'extracted_drugs'},
    key='label',
)

DISEASE_ENTITIES = StagedOutput(
    name='disease_entities',
    model=DISEASE_NER_MODEL,
    hub_models=(DISEASE_MODEL,),
    params={'labels': DISEASE_LABELS, 'min_length': DISEASE_MIN_LENGTH, 'drop_digits': DISEASE_DROP_DIGITS},
    logic_version=1,
    columns={'text': 'text', 'output': 'extracted_diseases'},
    key='text',
)

EXTRACTION = ExtractionOutput(
    name='extraction',
    model='aact_llm_extraction',
    params={'trial_fields': TRIAL_FIELDS},
    logic_version=1,
)
