import json

from vpo.cli import main


def test_cli_run(capsys):
    assert main(["run", "--rate", "50", "--duration", "5", "--admission", "shed_batch"]) == 0
    assert json.loads(capsys.readouterr().out)["completed"] > 0


def test_cli_report(tmp_path):
    assert main(["report", "--out", str(tmp_path), "--only", "failures"]) == 0
    rows = json.loads((tmp_path / "failures.json").read_text())
    assert [r["label"] for r in rows] == ["max_attempts=1", "max_attempts=3"]
