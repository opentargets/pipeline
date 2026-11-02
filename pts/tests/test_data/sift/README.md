# SIFT fixture (Ensembl 116)

Real rows of the Ensembl release 116 files, for the canonical translations of
ENST00000354588, ENST00000524958 (variant 11_62664853_G_A) and ENST00000390321,
ENST00000526893 (variant 22_22895477_C_A):

- `protein_function_predictions.txt.gz`: every row of
  `mysql/homo_sapiens_variation_116_38/protein_function_predictions.txt.gz` whose
  `translation_md5_id` is 20623, 29674, 64394 or 67384 (all analyses), byte for byte.
- `translation_md5.txt.gz`: the rows of `translation_md5.txt.gz` for those ids.
- `attrib.txt.gz`: `attrib.txt.gz`, whole (`sift` is attrib 267).
- `Homo_sapiens.GRCh38.pep.all.fa.gz`: the four records of `fasta/homo_sapiens/pep/Homo_sapiens.GRCh38.pep.all.fa.gz`.
