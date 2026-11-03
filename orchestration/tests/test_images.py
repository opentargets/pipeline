"""Tests for resolving every image a DAG config pins."""

from pathlib import Path

import pytest

from orchestration.utils import images


def write_config(tmp_path: Path) -> Path:
    path = tmp_path / 'dag.yaml'
    path.write_text("pis_version: '1.0'\npts_version: '2.0'\n")
    return path


def test_image_refs_expands_pts_to_both_images():
    assert images.image_refs({'pis': '1.0', 'pts': '2.0'}, 'r') == ['r/pis:1.0', 'r/pts:2.0', 'r/pts-nlp:2.0']


def test_main_reports_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(images, 'resolve', lambda ref: None if ref.endswith('/pts-nlp:2.0') else 'sha256:abc')

    assert images.main([str(write_config(tmp_path))]) == 1
    assert 'pts-nlp:2.0' in capsys.readouterr().out


def test_main_succeeds_when_every_image_resolves(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(images, 'resolve', lambda ref: 'sha256:abc')

    assert images.main([str(write_config(tmp_path))]) == 0


def test_resolve_reports_gcloud_errors(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    failed = images.subprocess.CompletedProcess([], 1, stdout='', stderr='ERROR: Reauthentication failed.')
    monkeypatch.setattr(images.subprocess, 'run', lambda *args, **kwargs: failed)

    assert images.resolve('r/pts:2.0') is None
    assert 'Reauthentication failed' in capsys.readouterr().err


def test_main_fails_on_a_config_without_versions(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    path = tmp_path / 'dag.yaml'
    path.write_text("run_name: 'x'\n")

    assert images.main([str(path)]) == 1
    assert 'no pis_version or pts_version' in capsys.readouterr().out
