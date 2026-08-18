"""Tests for the Google ADK adapter (M2).

Everything here is offline: constructing `google.adk` `Agent` objects
requires no network access or API key -- `TAVILY_API_KEY` is only checked
for *presence*, via `monkeypatch.setenv`, never actually used to call
anything.

`pytest.importorskip` at module scope means this whole file (and its
assertions about installed-ADK behavior) is skipped, not failed, when
`google-adk` isn't installed -- the core suite must stay green either way.
"""

import yaml
import pytest

pytest.importorskip("google.adk")

import commonadk  # noqa: E402  (import after importorskip, deliberately)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tavily_env(monkeypatch):
    """Satisfy researcher's one *required* env var; POSTGRES_DSN stays unset
    on purpose -- it's declared optional and must never block a build."""
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


@pytest.fixture()
def multi_parent_project(tmp_project, tavily_env):
    """The example, with a `coordinator -> writer` delegate edge added back
    on top of its shipped `coordinator -> researcher -> writer` tree, so
    `writer` becomes reachable from two parents again. `tmp_project` is a
    fresh copy of the shipped (now clean-tree) example -- kept as an
    in-test fixture purely to exercise the multi-parent rejection path,
    since the shipped example itself must build cleanly (plan.md v1
    intersection rule / M3 hypothesis test: the same `common/` folder has to
    run unmodified on Google ADK).
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


# ---------------------------------------------------------------------------
# sub_agents tree / multi-parent semantics
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env):
    """The shipped research-crew example is a clean tree --
    coordinator -delegate-> researcher -handoff-> writer, with no direct
    coordinator -> writer edge -- so it must build end-to-end unmodified.
    This is the M3 hypothesis test's entry point: the same `common/` folder
    has to build cleanly on every v1 target.
    """
    project = commonadk.load(example_common_dir)
    agent = project.build("coordinator", target="google-adk")

    assert agent.name == "coordinator"
    assert agent.instruction.strip() != ""
    assert [a.name for a in agent.sub_agents] == ["researcher"]

    researcher = agent.sub_agents[0]
    assert researcher.instruction.strip() != ""
    assert [a.name for a in researcher.sub_agents] == ["writer"]

    writer = researcher.sub_agents[0]
    assert writer.instruction.strip() != ""
    assert writer.sub_agents == []


def test_researcher_build_is_a_clean_tree(example_common_dir, tavily_env):
    """Building from `researcher` directly (not through `coordinator`) is
    also a clean tree -- researcher -> writer (handoff) -- and must succeed,
    carrying writer as its own sub_agent.
    """
    project = commonadk.load(example_common_dir)
    agent = project.build("researcher", target="google-adk")

    assert agent.name == "researcher"
    assert agent.instruction.strip() != ""
    assert [a.name for a in agent.sub_agents] == ["writer"]
    assert agent.sub_agents[0].instruction.strip() != ""


def test_multi_parent_graph_is_rejected(multi_parent_project):
    """Reintroducing a `coordinator -> writer` edge on top of the existing
    `researcher -> writer` handoff (in-test fixture only -- see
    `multi_parent_project`) makes `writer` reachable from two parents again.

    Installed google-adk (2.7.1) semantics: `BaseAgent.model_post_init` ->
    `__set_parent_agent_for_sub_agents` raises if a sub-agent *instance*
    already has a `parent_agent` -- but that guard only catches a shared
    instance, and naively building this tree would construct two *separate*
    `writer` instances (once under `researcher`, once directly under
    `coordinator`), each with exactly one parent, sailing right past ADK's
    check and silently duplicating the agent. So the adapter does its own
    reachability bookkeeping and must raise before ever constructing the
    duplicate -- this is that behavior.
    """
    with pytest.raises(ValueError) as exc_info:
        multi_parent_project.build("coordinator", target="google-adk")

    message = str(exc_info.value)
    assert "writer" in message
    assert "two different parents" in message
    assert "coordinator" in message and "researcher" in message


def test_cycle_in_reachable_graph_is_rejected(tmp_project, tavily_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop) must raise a clear error rather than recursing forever or
    building a broken tree.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "delegate"},
        {"from": "researcher", "to": "writer", "type": "handoff"},
        {"from": "writer", "to": "coordinator", "type": "handoff"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    with pytest.raises(ValueError, match="cycle"):
        project.build("coordinator", target="google-adk")


# ---------------------------------------------------------------------------
# model routing
# ---------------------------------------------------------------------------


def test_gemini_alias_resolves_to_bare_native_model_id(example_common_dir, tavily_env):
    """writer's model is the `fast` alias -> gemini/gemini-2.5-flash, and it
    has no per-target override, so the adapter must pass the BARE model id
    natively rather than wrapping it in LiteLlm.
    """
    project = commonadk.load(example_common_dir)
    agent = project.build("writer", target="google-adk")

    assert agent.model == "gemini-2.5-flash"


def test_non_gemini_model_is_litellm_wrapped(tmp_project, tavily_env):
    """A non-gemini provider (anthropic/claude-sonnet-5, via the `smart`
    alias) must be wrapped in google.adk.models.lite_llm.LiteLlm, carrying
    the FULL LiteLLM-format string.
    """
    from google.adk.models.lite_llm import LiteLlm

    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "smart"  # config.yaml alias: anthropic/claude-sonnet-5
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="google-adk")

    assert isinstance(agent.model, LiteLlm)
    assert agent.model.model == "anthropic/claude-sonnet-5"


def test_per_target_override_wins(example_common_dir, tavily_env):
    """researcher's base model is gemini/gemini-2.5-pro, but its
    `targets.google-adk.model` override (gemini-2.5-flash) must win and is
    passed through as-is (already SDK-native form).
    """
    project = commonadk.load(example_common_dir)
    agent = project.build("researcher", target="google-adk")

    assert agent.model == "gemini-2.5-flash"


# ---------------------------------------------------------------------------
# env preflight
# ---------------------------------------------------------------------------


def test_missing_required_env_var_blocks_build(example_common_dir, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    with pytest.raises(OSError) as exc_info:
        project.build("researcher", target="google-adk")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "Search API key used by search_web" in message
    # POSTGRES_DSN is declared `required: false` -- its absence must not be
    # reported as a blocking problem.
    assert "POSTGRES_DSN" not in message


def test_optional_env_var_absence_does_not_block(example_common_dir, monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    agent = project.build("researcher", target="google-adk")  # must not raise
    assert agent.name == "researcher"


def test_env_preflight_checks_agents_reachable_via_edges(tmp_project, monkeypatch):
    """Building `coordinator` must also check `researcher`'s env
    requirements, since researcher is reachable from coordinator via a
    delegate edge -- even though coordinator has no `requires.env` of its
    own. (`tmp_project` ships as a clean tree, so the multi-parent error
    doesn't mask this.)
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(tmp_project)

    with pytest.raises(OSError, match="TAVILY_API_KEY"):
        project.build("coordinator", target="google-adk")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------


def test_unknown_target_names_known_targets(example_common_dir, tavily_env):
    project = commonadk.load(example_common_dir)

    with pytest.raises(ValueError) as exc_info:
        project.build("coordinator", target="not-a-real-sdk")

    message = str(exc_info.value)
    assert "not-a-real-sdk" in message
    assert "google-adk" in message


def test_get_adapter_missing_sdk_gives_install_hint(monkeypatch):
    """Simulate google-adk not being installed by making the adapter
    registry's import of `commonadk.adapters.google_adk` fail, and check
    `get_adapter` turns that into a clear `pip install "commonadk[google]"`
    hint rather than a bare ImportError/traceback.
    """
    import importlib

    import commonadk.adapters as adapters_pkg

    real_import_module = importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "commonadk.adapters.google_adk":
            raise ImportError("No module named 'google'")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(adapters_pkg, "import_module", fake_import_module)

    with pytest.raises(ImportError) as exc_info:
        adapters_pkg.get_adapter("google-adk")

    message = str(exc_info.value)
    assert 'pip install "commonadk[google]"' in message
