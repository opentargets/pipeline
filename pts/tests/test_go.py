"""Tests for the go transformer."""

from __future__ import annotations

from pathlib import Path

from otter.config.model import Config

from pts.transformers.go import go
from pts.transformers.utils.dataset import scan_dataset

OBO = """format-version: 1.2

[Term]
id: GO:0000001
name: root
namespace: biological_process

[Term]
id: GO:0000002
name: parent
namespace: biological_process
is_a: GO:0000001

[Term]
id: GO:0000003
name: child
namespace: biological_process
relationship: part_of GO:0000002
relationship: has_part GO:0000004
relationship: regulates GO:0000005

[Term]
id: GO:0000004
name: component
namespace: biological_process

[Term]
id: GO:0000005
name: regulated
namespace: biological_process
"""


def test_ancestors_follow_only_is_a_and_part_of(tmp_path: Path) -> None:
    source = tmp_path / 'go.obo'
    source.write_text(OBO)
    go(str(source), str(tmp_path / 'go'), {}, Config(step='go', steps=['go'], work_path=tmp_path))
    rows = {row['id']: row for row in scan_dataset(str(tmp_path / 'go')).collect().iter_rows(named=True)}
    assert rows['GO:0000003']['ancestors'] == ['GO:0000001', 'GO:0000002']
    assert rows['GO:0000001']['descendants'] == ['GO:0000002', 'GO:0000003']
    assert rows['GO:0000004']['descendants'] == []
    assert rows['GO:0000005']['descendants'] == []
