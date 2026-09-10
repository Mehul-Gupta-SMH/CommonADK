"""Tests for `commonadk import` -- SKILL.md interoperability, part 2.

Every test here invokes `cli.main(argv)` in-process (no subprocess, no
network), matching test_cli_new.py's style. Fixtures live under
tests/fixtures/ (small, hand-authored SKILL.md files -- see FIXTURES below;
none of this is copied from any real SKILL.md library).
"""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from commonadk import cli
from commonadk.loader import load

FIXTURES = Path(__file__).resolve().parent / "fixtures"
SKILLS_NESTED = FIXTURES / "skills_nested"  # <name>/SKILL.md, Spotify's layout
SKILLS_FLAT = FIXTURES / "skills_flat"  # flat *.md
SKILLS_NEEDS_NORMALIZATION = FIXTURES / "skills_needs_normalization"
SKILLS_COLLISION = FIXTURES / "skills_collision"


# ---------------------------------------------------------------------------
# both directory layouts
# ---------------------------------------------------------------------------


def test_import_nested_layout(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    captured = capsys.readouterr()

    assert rc == 0
    assert (common_dir / "planner").is_dir()
    assert (common_dir / "archivist").is_dir()

    planner_cfg = yaml.safe_load((common_dir / "planner" / "agent-config.yaml").read_text())
    assert planner_cfg["name"] == "planner"
    assert "ordered list" in planner_cfg["description"]
    assert planner_cfg["tools"] == []
    assert planner_cfg["requires"] == {"env": []}

    # skill.md is the original file, frontmatter and all -- byte for byte.
    original = (SKILLS_NESTED / "planner" / "SKILL.md").read_text()
    assert (common_dir / "planner" / "skill.md").read_text() == original

    assert "planner" in captured.out
    assert "archivist" in captured.out


def test_import_flat_layout(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_FLAT), str(common_dir)])
    capsys.readouterr()

    assert rc == 0
    assert (common_dir / "summarizer").is_dir()
    assert (common_dir / "translator").is_dir()

    # translator.md has no frontmatter at all: name comes from the filename,
    # description falls back to "".
    translator_cfg = yaml.safe_load((common_dir / "translator" / "agent-config.yaml").read_text())
    assert translator_cfg["name"] == "translator"
    assert translator_cfg["description"] == ""

    summarizer_cfg = yaml.safe_load((common_dir / "summarizer" / "agent-config.yaml").read_text())
    assert summarizer_cfg["name"] == "summarizer"
    assert "summary" in summarizer_cfg["description"]


# ---------------------------------------------------------------------------
# output passes `commonadk validate` immediately
# ---------------------------------------------------------------------------


def test_import_output_passes_validate(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc_import = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc_import == 0
    capsys.readouterr()

    rc_validate = cli.main(["validate", str(common_dir)])
    captured = capsys.readouterr()

    assert rc_validate == 0
    assert "planner" in captured.out
    assert "archivist" in captured.out


def test_import_flat_output_passes_validate(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc_import = cli.main(["import", str(SKILLS_FLAT), str(common_dir)])
    assert rc_import == 0
    capsys.readouterr()

    rc_validate = cli.main(["validate", str(common_dir)])
    assert rc_validate == 0


def test_import_output_loads_with_real_loader(tmp_path):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc == 0

    project = load(common_dir)
    assert set(project.agents) == {"planner", "archivist"}
    assert project.graph.edges == []  # no invented edges


# ---------------------------------------------------------------------------
# --entry: honored, and the deterministic default
# ---------------------------------------------------------------------------


def test_import_entry_defaults_to_alphabetically_first(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    captured = capsys.readouterr()

    assert rc == 0
    # "archivist" < "planner" alphabetically.
    project = load(common_dir)
    assert project.config.entry == "archivist"
    assert "archivist" in captured.out
    assert "chosen automatically" in captured.out


def test_import_entry_is_honored(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir), "--entry", "planner"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "chosen automatically" not in captured.out
    project = load(common_dir)
    assert project.config.entry == "planner"
    assert project.graph.entry == "planner"


def test_import_unknown_entry_is_rejected(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir), "--entry", "nonexistent"])
    captured = capsys.readouterr()

    assert rc != 0
    assert "nonexistent" in captured.err
    assert not common_dir.exists() or not any(common_dir.iterdir())


# ---------------------------------------------------------------------------
# name normalization
# ---------------------------------------------------------------------------


def test_import_normalizes_name_and_rewrites_frontmatter(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_NEEDS_NORMALIZATION), str(common_dir)])
    captured = capsys.readouterr()

    assert rc == 0
    agent_dir = common_dir / "weird-name"
    assert agent_dir.is_dir()

    cfg = yaml.safe_load((agent_dir / "agent-config.yaml").read_text())
    assert cfg["name"] == "weird-name"

    # frontmatter `name` had to be rewritten to agree with the folder --
    # feature 1's own validation requires that agreement -- but everything
    # else in the file (description, body) survives untouched.
    skill_text = (agent_dir / "skill.md").read_text()
    assert "name: weird-name" in skill_text
    assert "name: Weird Name" not in skill_text
    assert "Demonstrate name normalization end to end." in skill_text
    assert "frontmatter name rewritten" in captured.out

    # and the result validates cleanly despite the rewrite.
    rc_validate = cli.main(["validate", str(common_dir)])
    assert rc_validate == 0


def test_import_dedupes_colliding_normalized_names(tmp_path):
    common_dir = tmp_path / "common"
    rc = cli.main(["import", str(SKILLS_COLLISION), str(common_dir)])

    assert rc == 0
    # one.md sorts before two.md -- deterministic suffixing.
    assert (common_dir / "shared").is_dir()
    assert (common_dir / "shared-2").is_dir()

    project = load(common_dir)
    assert {"shared", "shared-2"} <= set(project.agents)


# ---------------------------------------------------------------------------
# refuse-to-clobber behavior
# ---------------------------------------------------------------------------


def test_import_into_nonexistent_dir_creates_it(tmp_path):
    common_dir = tmp_path / "does" / "not" / "exist" / "yet"
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc == 0
    assert common_dir.is_dir()


def test_import_into_empty_existing_dir_is_fine(tmp_path):
    common_dir = tmp_path / "common"
    common_dir.mkdir()
    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc == 0
    assert (common_dir / "planner").is_dir()


def test_import_refuses_nonempty_non_project_dir(tmp_path, capsys):
    common_dir = tmp_path / "common"
    common_dir.mkdir()
    (common_dir / "random.txt").write_text("not a commonadk project\n")

    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    captured = capsys.readouterr()

    assert rc != 0
    assert "refusing to import" in captured.err
    # nothing was touched.
    assert not (common_dir / "planner").exists()
    assert (common_dir / "random.txt").exists()


def test_import_extends_a_valid_existing_project(tmp_path, capsys):
    common_dir = tmp_path / "common"
    rc_first = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc_first == 0
    capsys.readouterr()

    entry_before = load(common_dir).config.entry

    rc_second = cli.main(["import", str(SKILLS_FLAT), str(common_dir)])
    captured = capsys.readouterr()

    assert rc_second == 0
    assert "Extended" in captured.out
    project = load(common_dir)
    assert set(project.agents) == {"planner", "archivist", "summarizer", "translator"}
    # extending never changes the existing entry unless --entry says so.
    assert project.config.entry == entry_before

    rc_validate = cli.main(["validate", str(common_dir)])
    assert rc_validate == 0


def test_import_extend_never_overwrites_existing_agent(tmp_path):
    common_dir = tmp_path / "common"
    rc_first = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc_first == 0

    skill_before = (common_dir / "planner" / "skill.md").read_text()

    # re-importing the same nested dir must not clobber "planner" -- the
    # second copy is deduped to "planner-2" instead.
    rc_second = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    assert rc_second == 0

    assert (common_dir / "planner" / "skill.md").read_text() == skill_before
    assert (common_dir / "planner-2").is_dir()


def test_import_common_dir_is_a_file_is_rejected(tmp_path, capsys):
    common_dir = tmp_path / "common"
    common_dir.write_text("i am a file, not a directory\n")

    rc = cli.main(["import", str(SKILLS_NESTED), str(common_dir)])
    captured = capsys.readouterr()

    assert rc != 0
    assert "not a directory" in captured.err


# ---------------------------------------------------------------------------
# error paths
# ---------------------------------------------------------------------------


def test_import_missing_skills_dir_exits_nonzero(tmp_path, capsys):
    rc = cli.main(["import", str(tmp_path / "does-not-exist"), str(tmp_path / "common")])
    captured = capsys.readouterr()

    assert rc != 0
    assert "not found" in captured.err


def test_import_empty_skills_dir_exits_nonzero(tmp_path, capsys):
    empty_skills = tmp_path / "empty-skills"
    empty_skills.mkdir()

    rc = cli.main(["import", str(empty_skills), str(tmp_path / "common")])
    captured = capsys.readouterr()

    assert rc != 0
    assert "no SKILL.md files found" in captured.err
