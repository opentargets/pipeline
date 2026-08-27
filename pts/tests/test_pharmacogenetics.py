"""Tests for the pharmacogenetics module's OpenAI failure handling.

The step annotates phenotypes from a curated lookup table and calls the OpenAI API only
for text the table does not cover. That call is against a third party, from a Dataproc
driver, one request per text — so it fails sometimes, and what it does when it fails
decides whether a run survives.
"""

from unittest.mock import MagicMock

import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from pts.pyspark.pharmacogenetics import parse_phenotype_with_gpt, parse_phenotypes


def _client(side_effect=None, content: str | None = None) -> MagicMock:
    """An OpenAI client stub whose `create` either raises or returns `content`."""
    client = MagicMock()
    if side_effect is not None:
        client.responses.create.side_effect = side_effect
    else:
        completion = MagicMock()
        completion.output_text = content
        completion.output = [MagicMock(content=[MagicMock(text=content)])]
        client.responses.create.return_value = completion
    return client


def _connection_error() -> APIConnectionError:
    return APIConnectionError(request=MagicMock())


class TestParsePhenotypeWithGpt:
    def test_a_connection_error_returns_none_instead_of_raising(self) -> None:
        """The exact failure that killed run do/platform-2608-2.

        `APIConnectionError` propagated out of the step, discarding 115s of completed
        work and blocking ~200 downstream tasks. The text is simply left unextracted
        instead, which is the same state it was in before the API was consulted.
        """
        assert parse_phenotype_with_gpt('some genotype text', _client(side_effect=_connection_error())) is None

    @pytest.mark.parametrize(
        'error',
        [
            APITimeoutError(request=MagicMock()),
            RateLimitError('rate limited', response=MagicMock(status_code=429), body=None),
        ],
        ids=['timeout', 'rate_limit'],
    )
    def test_other_api_failures_are_also_survivable(self, error: Exception) -> None:
        """A key that starts rate limiting mid-run must degrade, not abort."""
        assert parse_phenotype_with_gpt('some genotype text', _client(side_effect=error)) is None

    def test_a_successful_extraction_is_returned(self) -> None:
        client = _client(content='{"gptExtractedPhenotype": ["increased response"]}')
        assert parse_phenotype_with_gpt('text', client) == ['increased response']

    def test_unparseable_content_returns_none(self) -> None:
        """Pre-existing behaviour, pinned so the new try block above it cannot swallow it."""
        assert parse_phenotype_with_gpt('text', _client(content='not json at all')) is None


class TestParsePhenotypes:
    def test_every_text_failing_still_produces_a_frame(self, spark) -> None:
        """A total outage yields an empty result, not an exception.

        Rows whose text could not be extracted keep a null phenotypeText and flow on. The
        step degrades; it does not take the run down with it.
        """
        session = MagicMock()
        session.spark = spark
        result = parse_phenotypes(session, ['a', 'b', 'c'], _client(side_effect=_connection_error()))
        assert result.count() == 0
        assert result.columns == ['genotypeAnnotationText', 'phenotypeText']

    def test_a_partial_outage_keeps_what_succeeded(self, spark) -> None:
        """One failure must not discard the extractions that worked."""
        good = MagicMock()
        good.output_text = '{"gptExtractedPhenotype": ["increased response"]}'
        good.output = [MagicMock(content=[MagicMock(text='{"gptExtractedPhenotype": ["increased response"]}')])]
        client = MagicMock()
        client.responses.create.side_effect = [good, _connection_error(), good]

        session = MagicMock()
        session.spark = spark
        result = parse_phenotypes(session, ['a', 'b', 'c'], client)
        assert result.count() == 2, 'a single failed call discarded the successful ones'
        assert sorted(r.genotypeAnnotationText for r in result.collect()) == ['a', 'c']


class TestResponseShape:
    def test_a_missing_key_is_not_treated_as_an_empty_extraction(self) -> None:
        """A model answering in the wrong shape must be retried, not recorded as done.

        `gptExtractedPhenotype` absent means the model ignored the requested shape. That
        used to default to `[]`, which `parse_phenotypes` stores as a successful
        extraction — so the text entered the lookup table permanently empty and was never
        attempted again. Observed while probing a model swap: both models answered with
        `phenotype` and `phenotypes` when the prompt was abbreviated.
        """
        client = _client(content='{"phenotype": "increased response"}')
        assert parse_phenotype_with_gpt('text', client) is None

    def test_a_genuinely_empty_extraction_is_kept(self) -> None:
        """An empty list is real data, not a failure.

        1,411 of the curated entries have an empty phenotypeText because the annotation
        describes no phenotype. Those must stay distinguishable from a bad response, or
        the fix above would make the step re-query them forever.
        """
        client = _client(content='{"gptExtractedPhenotype": []}')
        assert parse_phenotype_with_gpt('text', client) == []
