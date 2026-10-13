import pytest

from pps.tasks.pyspark import session_kwargs


def test_session_kwargs_default_to_a_local_master(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv('PYSPARK_GATEWAY_PORT', raising=False)

    assert session_kwargs({'write_mode': 'overwrite'}) == {
        'spark_uri': 'local[*]',
        'app_name': 'pps',
        'write_mode': 'overwrite',
    }


def test_session_kwargs_default_to_yarn_under_spark_submit(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYSPARK_GATEWAY_PORT', '12345')

    assert session_kwargs({})['spark_uri'] == 'yarn'


def test_session_kwargs_keep_an_explicit_master(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv('PYSPARK_GATEWAY_PORT', '12345')

    assert session_kwargs({'spark_uri': 'local[2]'})['spark_uri'] == 'local[2]'


def test_session_kwargs_reject_unknown_settings():
    with pytest.raises(ValueError, match='invalid session settings'):
        session_kwargs({'start_hial': True})
