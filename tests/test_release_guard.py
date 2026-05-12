from __future__ import annotations

from spare_molgen.release_guard import release_check


def test_release_check_flags_tensor_artifacts(tmp_path):
    (tmp_path / "model.pt").write_bytes(b"not a real checkpoint")

    findings = release_check(tmp_path)

    assert any("model.pt" in finding for finding in findings)


def test_release_check_accepts_source_tree(tmp_path):
    src = tmp_path / "src" / "pkg"
    src.mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")

    assert release_check(tmp_path) == []


def test_release_check_ignores_local_environment(tmp_path):
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "model.pt").write_bytes(b"local dev environment")

    assert release_check(tmp_path) == []


def test_release_check_flags_private_data_path(tmp_path):
    private = tmp_path / "data" / "private"
    private.mkdir(parents=True)
    (private / "train.jsonl").write_text('{"text": "private"}\n', encoding="utf-8")

    findings = release_check(tmp_path)

    assert any("data/private/train.jsonl" in finding for finding in findings)


def test_release_check_flags_local_user_path(tmp_path):
    (tmp_path / "README.md").write_text(
        "local path: " + "/" + "Users/example/internal/train.jsonl\n",
        encoding="utf-8",
    )

    findings = release_check(tmp_path)

    assert any("local user path" in finding for finding in findings)
