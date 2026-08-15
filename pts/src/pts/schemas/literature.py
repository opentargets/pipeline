"""Schema for the literature (EPMC) export, as read by the publication lookup table.

Pins only the columns that table needs, and pins them deliberately rather than for tidiness:
polars infers a newline delimited json column's dtype from a sample of the LEADING rows, and
`pmid`/`pmcid` are absent from the first rows of some parts of the real export. Inferred that way
they come out `Null`, and the read then either fails or -- under `ignore_errors` -- silently
discards every value in the column: 386,627 pmids in one of the 26.06 export's 56 parts.

Pinning makes the read independent of where the nulls happen to fall, which is also what lets the
parts be read separately.
"""

import polars as pl

literature_schema = {
    'source': pl.String,
    'pmid': pl.String,
    'id': pl.String,
    'pmcid': pl.String,
    'firstPublicationDate': pl.String,
}
