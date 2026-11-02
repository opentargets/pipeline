"""Finds the consolidated New Drugs PDF on the PMDA page and downloads it once."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Self
from urllib.parse import urljoin, urlparse

import requests
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pis.fetch import FetchError, fetch_once, is_fetched, validate_destination

HEADING = re.compile(r'h[1-6]')
SECTION = 'New Drugs'
CONSOLIDATED_PREFIXES = ('April 2004 to', 'All')
SIZE_SUFFIX = re.compile(r'\s*\[[^\]]*\]\s*$')


@dataclass(frozen=True)
class PmdaLink:
    """A PDF link found on the PMDA page."""

    url: str
    text: str


class _NewDrugsParser(HTMLParser):
    """Collects the PDF links between the `New Drugs` heading and the next heading without a link."""

    def __init__(self, page_url: str) -> None:
        super().__init__()
        self.page_url = page_url
        self.links: list[PmdaLink] = []
        self._in_section = False
        self._done = False
        self._heading: list[str] | None = None
        self._heading_has_link = False
        self._href: str | None = None
        self._anchor: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if HEADING.fullmatch(tag):
            self._heading = []
            self._heading_has_link = False
        elif tag == 'a':
            self._href = dict(attrs).get('href')
            self._anchor = []
            if self._heading is not None:
                self._heading_has_link = True

    def handle_data(self, data: str) -> None:
        if self._heading is not None:
            self._heading.append(data)
        if self._href is not None:
            self._anchor.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == 'a' and self._href is not None:
            self._end_anchor()
        elif HEADING.fullmatch(tag) and self._heading is not None:
            self._end_heading()

    def _end_anchor(self) -> None:
        href, self._href = self._href, None
        if not self._in_section or self._done or not href:
            return
        if urlparse(href).path.lower().endswith('.pdf'):
            self.links.append(PmdaLink(urljoin(self.page_url, href), _squash(''.join(self._anchor))))

    def _end_heading(self) -> None:
        text = _squash(''.join(self._heading or []))
        self._heading = None
        if self._heading_has_link:
            return
        if self._in_section:
            self._done = True
        elif text == SECTION:
            self._in_section = True


def _squash(text: str) -> str:
    return ' '.join(text.split())


def find_new_drugs_pdf(html: str, page_url: str) -> PmdaLink:
    """Find the single consolidated PDF under the `New Drugs` heading.

    Candidates are the PDF links after the heading and before the next heading
    that is not itself a link. A candidate matches when its text starts with
    `April 2004 to` or `All`.

    Args:
        html: The PMDA page.
        page_url: The URL of the page, which relative links are joined against.

    Returns:
        The matching link.

    Raises:
        FetchError: If zero or more than one candidate matches. The message
            names the page URL and every candidate.
    """
    parser = _NewDrugsParser(page_url)
    parser.feed(html)
    parser.close()
    matches = [link for link in parser.links if link.text.startswith(CONSOLIDATED_PREFIXES)]
    if len(matches) != 1:
        found = '; '.join(f'{link.text} ({link.url})' for link in parser.links) or 'no PDF links'
        raise FetchError(f'{page_url} has {len(matches)} consolidated New Drugs PDF links, expected 1; found: {found}')
    return matches[0]


class FetchPmdaSpec(Spec):
    """Configuration fields for the fetch PMDA task.

    This task has the following custom configuration fields:
        - page (str): The URL of the PMDA approved products page.
        - destination (str): The path, relative to `release_uri`, to write to.
        - as_of (str): The staging date, `YYYY-MM-DD`.
    """

    page: str
    destination: str
    as_of: str


class FetchPmda(Task):
    """Download the consolidated New Drugs PDF from the PMDA page once per staging date.

    If the destination already exists it is kept and the page is not fetched.
    The sidecar file beside the destination records the period the PDF covers.

    .. note:: `destination` will be prepended with either
        :py:obj:`otter.config.model.Config.release_uri` or
        :py:obj:`otter.config.model.Config.work_path`.
    """

    def __init__(self, spec: FetchPmdaSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: FetchPmdaSpec

    @report
    def run(self) -> Self:
        config = self.context.config
        if is_fetched(self.spec.destination, config):
            fetch_once(self.spec.page, self.spec.destination, config, self.spec.as_of)
            return self

        response = requests.get(self.spec.page, timeout=300)
        if not response.ok:
            raise FetchError(f'{self.spec.page} returned status {response.status_code}')
        link = find_new_drugs_pdf(response.text, self.spec.page)
        fetch_once(
            link.url,
            self.spec.destination,
            config,
            self.spec.as_of,
            extra={'period': SIZE_SUFFIX.sub('', link.text), 'page': self.spec.page},
        )
        return self

    @report
    def validate(self) -> Self:
        """Check that the destination exists and is not empty."""
        validate_destination(self.spec.destination, self.context.config)
        return self
