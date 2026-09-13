"""Publication rejects oversized or missing files before uploading anything."""

from pathlib import Path

from scripts.check_distribution_size import main


def test_files_at_the_limit_pass(tmp_path, capsys):
    wheel = tmp_path / "example.whl"
    sdist = tmp_path / "example.tar.gz"
    wheel.write_bytes(b"x" * 10)
    sdist.write_bytes(b"x" * 9)
    assert main(["--limit-bytes", "10", str(wheel), str(sdist)]) == 0
    assert "10 / 10 bytes" in capsys.readouterr().out


def test_one_oversized_file_rejects_the_batch(tmp_path, capsys):
    wheel = tmp_path / "example.whl"
    sdist = tmp_path / "example.tar.gz"
    wheel.write_bytes(b"x")
    sdist.write_bytes(b"x" * 11)
    assert main(["--limit-bytes", "10", str(wheel), str(sdist)]) == 1
    assert "exceeds" in capsys.readouterr().err


def test_missing_and_empty_files_fail_closed(tmp_path):
    empty = tmp_path / "empty.whl"
    empty.touch()
    assert main([str(empty)]) == 1
    assert main([str(tmp_path / "missing.tar.gz")]) == 1
    assert main([str(tmp_path)]) == 1


def test_workflow_checks_both_formats_before_uploading_artifacts():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/publish.yml").read_text()
    command = "python scripts/check_distribution_size.py dist/*.whl dist/*.tar.gz"
    assert command in workflow
    assert workflow.index(command) < workflow.index("name: Upload build artifacts")
    assert "needs: build" in workflow
