r"""Identifier shapes shared by the output dataset models.

Each is anchored, so it matches the whole value. A model uses the most specific shape every value of a
column was observed to match.
"""

#: Ensembl gene ID.
ENSEMBL_GENE_ID = r'^ENSG\d{11}$'

#: Ensembl transcript ID.
ENSEMBL_TRANSCRIPT_ID = r'^ENST\d{11}$'

#: Ensembl protein ID.
ENSEMBL_PROTEIN_ID = r'^ENSP\d{11}$'

#: sha1 hex digest.
SHA1_HEX = r'^[0-9a-f]{40}$'

#: md5 hex digest.
MD5_HEX = r'^[0-9a-f]{32}$'

#: ChEMBL identifier.
CHEMBL_ID = r'^CHEMBL\d+$'

#: dbSNP reference SNP identifier.
RS_ID = r'^rs\d+$'

#: Variant identifier: chromosome, position, reference and alternate allele.
VARIANT_ID = r'^(\d+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+$'

#: Variant identifier, or the `OTVAR_` hash replacing a long one.
VARIANT_ID_OR_HASH = r'^((\d+|X|Y|MT)_\d+_[ACGTN]+_[ACGTN]+|OTVAR_[0-9a-f]{32})$'

#: An ISO `YYYY-MM-DD` date.
ISO_DATE = r'^\d{4}-\d{2}-\d{2}$'

#: Digits only, e.g. a PubMed identifier.
DIGITS = r'^\d+$'

#: Ontology-style identifier: a namespace prefix, underscore, code -- `EFO_0000305`, `MONDO_0004975`.
ONTOLOGY_ID = r'^[A-Za-z]+_[A-Za-z0-9]+$'
