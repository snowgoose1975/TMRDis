import sys
import subprocess
from pathlib import Path
from types import SimpleNamespace
import pytest
from my_predictor import paper


def test_cli_returns_success_for_training_result_metadata(monkeypatch):
    fake = SimpleNamespace(train_main=lambda argv: {'status':'complete'})
    monkeypatch.setattr(paper,'import_module',lambda name:fake)
    assert paper.main(['train','--example']) == 0


def test_exported_configuration_has_all_ten_member_identities(tmp_path):
    out=tmp_path/'config'
    assert paper.main(['config','--output-dir',str(out)]) == 0
    rows=(out/'paper-v1-members.tsv').read_text().splitlines()
    assert len(rows)==11
    assert 'prott5-s20260926' in rows[1]
    with pytest.raises(FileExistsError):
        paper.main(['config','--output-dir',str(out)])


def test_replay_help_does_not_import_numpy_or_torch():
    root=Path(__file__).resolve().parents[1]
    process=subprocess.run([sys.executable,'-S','-m','my_predictor.paper','replay','--help'],cwd=root,text=True,capture_output=True)
    assert process.returncode==0,process.stderr
    assert '--artifact-root' in process.stdout
