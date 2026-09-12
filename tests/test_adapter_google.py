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
    """All three edges rewritten to `handoff` (`coordinator -> researcher ->
    writer` plus a direct `coordinator -> writer`), so `writer` is
    `handoff`-reachable from two parents within the SAME sub_agents tree.
    `tmp_project` is a fresh copy of the shipped example -- kept as an
    in-test fixture purely to exercise the multi-parent rejection path,
    since the shipped example itself must build cleanly (plan.md v1
    intersection rule / M3 hypothesis test: the same `common/` folder has to
    run unmodified on Google ADK).

    Since issue #10 (delegate/handoff distinction): only `handoff` edges
    join the sub_agents tree this constraint is about, and only within one
    continuous handoff-reachable scope -- if `coordinator -> researcher`
    stayed `delegate` (as shipped), `researcher` would be built as its own
    independent `AgentTool` subtree with fresh tree-tracking state, and the
    two paths to `writer` would never actually collide (see
    `test_delegate_edge_bypasses_the_sub_agents_tree_constraint` below,
    which exercises exactly that non-conflict). All three edges are
    `handoff` here so the conflict is genuine.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "handoff"},
        {"from": "researcher", "to": "writer", "type": "handoff"},
        {"from": "coordinator", "to": "writer", "type": "handoff"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


# ---------------------------------------------------------------------------
# sub_agents tree / multi-parent semantics
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env):
    """The shipped research-crew example -- coordinator -delegate->
    researcher -handoff-> writer -- must build end-to-end unmodified. This
    is the M3 hypothesis test's entry point: the same `common/` folder has
    to build cleanly on every v1 target.

    Since issue #10 (delegate/handoff distinction): coordinator's only edge
    is `delegate`, so coordinator.sub_agents must be EMPTY and researcher
    must instead be reachable through an `AgentTool` in coordinator.tools.
    researcher -> writer is still `handoff`, so writer stays a genuine
    sub_agent of researcher, unchanged.
    """
    from google.adk.tools import AgentTool

    project = commonadk.load(example_common_dir)
    agent = project.build("coordinator", target="google-adk")

    assert agent.name == "coordinator"
    assert agent.instruction.strip() != ""
    assert agent.sub_agents == []  # coordinator has no handoff edges

    researcher_tools = [t for t in agent.tools if isinstance(t, AgentTool)]
    assert [t.agent.name for t in researcher_tools] == ["researcher"]
    researcher = researcher_tools[0].agent

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


# ---------------------------------------------------------------------------
# delegate/handoff distinction (issue #10, first checkbox)
# ---------------------------------------------------------------------------


def test_delegate_edge_becomes_agent_tool_not_sub_agent(example_common_dir, tavily_env):
    """A `delegate` edge must produce an `AgentTool` in the source's `tools`
    list, never an entry in `sub_agents`."""
    from google.adk.tools import AgentTool

    project = commonadk.load(example_common_dir)
    agent = project.build("coordinator", target="google-adk")  # -delegate-> researcher

    assert agent.sub_agents == []
    assert any(isinstance(t, AgentTool) and t.agent.name == "researcher" for t in agent.tools)


def test_handoff_edge_becomes_sub_agent_not_agent_tool(example_common_dir, tavily_env):
    """A `handoff` edge must produce a `sub_agents` entry, never an
    `AgentTool` in `tools`."""
    from google.adk.tools import AgentTool

    project = commonadk.load(example_common_dir)
    agent = project.build("researcher", target="google-adk")  # -handoff-> writer

    assert [a.name for a in agent.sub_agents] == ["writer"]
    assert not any(isinstance(t, AgentTool) for t in agent.tools)


def test_delegate_edge_bypasses_the_sub_agents_tree_constraint(tmp_project, tavily_env):
    """Unlike a `handoff` edge (test_multi_parent_graph_is_rejected below),
    a `delegate` edge reaching an already-handoff-claimed destination must
    NOT be rejected: `writer` is already researcher's `handoff` sub_agent,
    and adding a `coordinator -> writer` DELEGATE edge on top must build
    successfully, wrapping a second, independent `writer` Agent instance in
    an AgentTool on coordinator -- no shared `parent_agent` state to
    conflict over (see module docstring, "Edge semantics").
    """
    from google.adk.tools import AgentTool

    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    project = commonadk.load(tmp_project)

    agent = project.build("coordinator", target="google-adk")  # must not raise

    delegate_targets = {t.agent.name for t in agent.tools if isinstance(t, AgentTool)}
    assert delegate_targets == {"researcher", "writer"}
    assert agent.sub_agents == []


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
# model_params
# ---------------------------------------------------------------------------


def test_full_model_params_land_on_generate_content_config(tmp_project, tavily_env):
    """Every mapped model_params key (see google_adk.py's module docstring
    and `_MODEL_PARAM_MAP`) must land on the built agent's
    `generate_content_config`, under the field name `GenerateContentConfig`
    actually declares -- `max_tokens` -> `max_output_tokens`, `stop` ->
    `stop_sequences`, and everything else passed straight through.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"] = {
        "temperature": 0.3,
        "max_tokens": 2048,
        "top_p": 0.9,
        "top_k": 40,
        "stop": ["END"],
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "seed": 42,
    }
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    agent = project.build("writer", target="google-adk")

    cfg = agent.generate_content_config
    assert cfg.temperature == 0.3
    assert cfg.max_output_tokens == 2048
    assert cfg.top_p == 0.9
    assert cfg.top_k == 40
    assert cfg.stop_sequences == ["END"]
    assert cfg.presence_penalty == 0.1
    assert cfg.frequency_penalty == 0.2
    assert cfg.seed == 42


def test_unsupported_model_params_key_is_warned_and_ignored(tmp_project, tavily_env):
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"]["not_a_real_param"] = 1
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match="model_params key 'not_a_real_param'"):
        agent = project.build("writer", target="google-adk")

    assert agent is not None  # build still succeeds


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
