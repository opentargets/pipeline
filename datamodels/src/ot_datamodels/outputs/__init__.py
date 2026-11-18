"""Models of the datasets PTS writes, one module per dataset (or family of datasets).

Every concrete `DatasetModel` defined anywhere in this package is discovered automatically by
`ot_datamodels.discovery` -- there is nothing to register.

A model states what holds for the data's meaning, not for one release's values: the next release must
not fail because a new value appeared.

* dtype and struct field order exactly as written;
* nullable wherever the data may be null (or NaN);
* numbers bounded only at natural limits -- `[0, 1]` for a probability or score, `>= 0` for a count --
  never at the observed min/max;
* strings: an identifier shape as a `pattern`; a `Literal` only for a closed vocabulary, listing every
  value the code can write, not only the values seen (`tests/test_schema_vocabularies.py` checks the
  vocabularies the code defines); otherwise `min_length=1` where an empty string is never valid;
* list lengths unconstrained, except that valid evidence carries no quality flag;
* descriptions, primary keys and references as in croissant, a primary key only where the data is
  unique on it.
"""
