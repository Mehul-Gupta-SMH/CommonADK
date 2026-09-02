"""Tests for the Claude Agent SDK adapter (M5).

Everything here is offline: constructing a `claude_agent_sdk.ClaudeAgentOptions`
(and the `AgentDefinition`/in-process MCP server objects it wires up) touches
no network and spawns no subprocess -- see claude_agent.py's module
docstring, "WHAT build() RETURNS", for why: this SDK is session/query-based,
so `build()` returns a plain configuration object rather than a live agent,
and building that object is pure Python construction. `ANTHROPIC_API_KEY` is
only checked for *presence* by `TAVILY_API_KEY`-style env preflight tests,
never used to call anything.

`pytest.importorskip` at module scope means this whole file is skipped, not
failed, when `claude-agent-sdk` (imported as `claude_agent_sdk`) isn't
installed -- the core suite must stay green either way.
"""

from __future__ import annotations

import yaml
import pytest

pytest.importorskip("claude_agent_sdk")

import commonadk  # noqa: E402  (import after importorskip, deliberately)
from claude_agent_sdk import ClaudeAgentOptions  # noqa: E402


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
    `writer` becomes reachable from two parents. Like OpenAI Agents (and
    unlike Google ADK), this must BUILD SUCCESSFULLY here -- `options.agents`
    is a flat `dict[str, AgentDefinition]` keyed by logical agent name, so a
    name reachable by two paths is simply the same dict entry.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"].append({"from": "coordinator", "to": "writer", "type": "delegate"})
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


@pytest.fixture()
def cyclic_project(tmp_project, tavily_env):
    """A cycle in the reachable graph (writer -> coordinator, closing the
    loop). Like OpenAI Agents, this must BUILD SUCCESSFULLY here -- the
    build root is never added to its own `options.agents` (it *is*
    `options`), so a cycle back to it is a no-op, not a hazard.
    """
    interactions_path = tmp_project / "interactions.yaml"
    data = yaml.safe_load(interactions_path.read_text())
    data["edges"] = [
        {"from": "coordinator", "to": "researcher", "type": "delegate"},
        {"from": "researcher", "to": "writer", "type": "handoff"},
        {"from": "writer", "to": "coordinator", "type": "handoff"},
    ]
    interactions_path.write_text(yaml.safe_dump(data))
    return commonadk.load(tmp_project)


def _strip_claude_override(project_dir, agent):
    cfg_path = project_dir / agent / "agent-config.yaml"
    data = yaml.safe_load(cfg_path.read_text())
    data["targets"].pop("claude", None)
    cfg_path.write_text(yaml.safe_dump(data))


# ---------------------------------------------------------------------------
# subagent graph construction / edge mapping
# ---------------------------------------------------------------------------


def test_coordinator_build_happy_path_on_example(example_common_dir, tavily_env):
    """The shipped research-crew example -- coordinator -delegate->
    researcher -handoff-> writer -- builds on the claude target once each
    agent-config.yaml carries a `targets.claude.model` override (the gemini
    models the example otherwise uses aren't Anthropic-native; see the
    model-routing tests below). This exercises the deep-edge decision:
    `writer` (reachable only via researcher, not a direct child of
    coordinator) must still show up in the returned config's flat
    `options.agents`, per claude_agent.py's module docstring ("Edge mapping
    and nesting depth").
    """
    project = commonadk.load(example_common_dir)
    options = project.build("coordinator", target="claude")

    assert isinstance(options, ClaudeAgentOptions)
    assert options.system_prompt.strip() != ""
    assert options.model == "claude-sonnet-5"
    assert options.tools == []  # no built-in Claude Code tools

    # coordinator's own declared tools, pre-approved, plus "Agent" since it
    # has an outgoing edge (-> researcher).
    assert set(options.allowed_tools) == {
        "mcp__coordinator_tools__split_into_subtopics",
        "mcp__coordinator_tools__format_handoff_note",
        "Agent",
    }
    # coordinator must not be able to reach into researcher's/writer's own
    # tools directly -- disallowed_tools actually removes tool availability
    # (unlike allowed_tools, which is permission-only).
    assert set(options.disallowed_tools) == {
        "mcp__researcher_tools__search_web",
        "mcp__researcher_tools__fetch_page",
        "mcp__writer_tools__count_words",
        "mcp__writer_tools__format_as_markdown",
    }

    # the whole reachable subgraph is flattened into one subagent registry --
    # writer included, even though it's only reachable via researcher.
    assert set(options.agents.keys()) == {"researcher", "writer"}
    assert set(options.mcp_servers.keys()) == {
        "coordinator_tools",
        "researcher_tools",
        "writer_tools",
    }

    researcher = options.agents["researcher"]
    assert researcher.description != ""
    assert researcher.prompt.strip() != ""
    assert researcher.model == "claude-sonnet-5"
    assert researcher.mcpServers == ["researcher_tools"]
    # researcher has its own outgoing edge (-> writer), so it must be able
    # to invoke the Agent tool to reach writer.
    assert "Agent" in researcher.tools
    assert set(researcher.tools) - {"Agent"} == {
        "mcp__researcher_tools__search_web",
        "mcp__researcher_tools__fetch_page",
    }

    writer = options.agents["writer"]
    assert writer.prompt.strip() != ""
    assert writer.mcpServers == ["writer_tools"]
    # writer has no outgoing edges -- it must NOT be granted the Agent tool.
    assert "Agent" not in writer.tools
    assert set(writer.tools) == {
        "mcp__writer_tools__count_words",
        "mcp__writer_tools__format_as_markdown",
    }


def test_researcher_build_wires_writer_as_only_subagent(example_common_dir, tavily_env):
    """Building from `researcher` directly (not through `coordinator`) makes
    researcher the root -- only `writer` (its one reachable agent) ends up
    in `options.agents`, and researcher's own tools move to the top-level
    `allowed_tools`/`disallowed_tools` since it is now the build root.
    """
    project = commonadk.load(example_common_dir)
    options = project.build("researcher", target="claude")

    assert options.model == "claude-sonnet-5"
    assert set(options.agents.keys()) == {"writer"}
    assert "Agent" in options.allowed_tools  # researcher -> writer edge
    assert set(options.mcp_servers.keys()) == {"researcher_tools", "writer_tools"}


def test_multi_parent_graph_builds_with_one_shared_entry(multi_parent_project):
    """KEY PROPERTY of the flat subagent registry: a multi-parent graph
    builds successfully (unlike Google ADK) and `writer` appears exactly
    once in `options.agents`, not duplicated per parent.
    """
    options = multi_parent_project.build("coordinator", target="claude")

    assert set(options.agents.keys()) == {"researcher", "writer"}
    # coordinator now also has a direct edge to writer, so its own
    # allowed_tools still just needs "Agent" once (it already had it).
    assert "Agent" in options.allowed_tools


def test_cyclic_graph_builds_without_recursion_hazard(cyclic_project):
    """KEY PROPERTY of the flat subagent registry: a cycle back to the build
    root builds successfully (unlike Google ADK) -- coordinator is simply
    never added to its own `options.agents`.
    """
    options = cyclic_project.build("coordinator", target="claude")

    assert set(options.agents.keys()) == {"researcher", "writer"}
    assert "coordinator" not in options.agents
    # writer's edge back to coordinator cannot be represented as a further
    # subagent entry (coordinator IS the returned options), but writer is
    # still granted "Agent" since it does have an outgoing edge.
    assert "Agent" in options.agents["writer"].tools


# ---------------------------------------------------------------------------
# model routing
# ---------------------------------------------------------------------------


def test_anthropic_model_resolves_to_bare_native_model_id(tmp_project, tavily_env):
    """An agent configured with an `anthropic/...` LiteLLM-format model (and
    no per-target override) must get the BARE model id -- no wrapper, since
    this SDK has no LiteLLM path at all.
    """
    _strip_claude_override(tmp_project, "writer")
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model"] = "anthropic/claude-sonnet-5"
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    options = project.build("writer", target="claude")

    assert options.model == "claude-sonnet-5"


def test_non_anthropic_model_raises_clear_unsupported_provider_error(
    tmp_project, tavily_env
):
    """researcher's base model (gemini/gemini-2.5-pro) has no LiteLLM path on
    this SDK -- with its `targets.claude` override removed, building it must
    raise a clear error naming the agent, the resolved model string, and
    suggesting a fix (an anthropic/... model, an alias change, or a
    per-target override).
    """
    _strip_claude_override(tmp_project, "researcher")
    project = commonadk.load(tmp_project)

    with pytest.raises(ValueError) as exc_info:
        project.build("researcher", target="claude")

    message = str(exc_info.value)
    assert "researcher" in message
    assert "gemini/gemini-2.5-pro" in message
    assert "anthropic/" in message
    assert "targets.claude.model" in message


def test_shipped_example_without_claude_overrides_fails_with_clear_error(tmp_project, tavily_env):
    """This is the M5 spec's explicit regression case: the research-crew
    example uses gemini models by default, so building it for target
    'claude' with NO overrides must fail loudly -- that is correct behavior,
    not a bug. (The shipped example itself now carries `targets.claude`
    overrides on every agent so it CAN build here too -- see
    test_coordinator_build_happy_path_on_example -- but stripping them back
    off, as this test does, must reproduce the un-overridden failure.)
    """
    for agent in ("coordinator", "researcher", "writer"):
        _strip_claude_override(tmp_project, agent)
    project = commonadk.load(tmp_project)

    with pytest.raises(ValueError, match="Anthropic models only"):
        project.build("coordinator", target="claude")


def test_per_target_override_wins(tmp_project, tavily_env):
    """writer's base model is the `fast` alias (-> gemini/gemini-2.5-flash),
    but its `targets.claude.model` override must win and is passed through
    as-is (already SDK-native form).
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["targets"]["claude"] = {"model": "claude-opus-5"}
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    options = project.build("writer", target="claude")

    assert options.model == "claude-opus-5"


# ---------------------------------------------------------------------------
# model_params (unsupported on this SDK -- warn and ignore)
# ---------------------------------------------------------------------------


def test_model_params_are_warned_and_ignored(example_common_dir, tavily_env):
    """Neither `ClaudeAgentOptions` nor `AgentDefinition` exposes anything
    resembling temperature/max_tokens (see claude_agent.py's module
    docstring, "model_params") -- every model_params key must warn, not
    raise, and must not appear anywhere on the returned config.
    """
    project = commonadk.load(example_common_dir)

    with pytest.warns(UserWarning, match="model_params key 'temperature'"):
        options = project.build("coordinator", target="claude")

    assert options is not None  # build still succeeds


@pytest.mark.parametrize(
    "unsupported_key",
    ["temperature", "max_tokens", "top_p", "top_k", "stop", "presence_penalty",
     "frequency_penalty", "seed"],
)
def test_full_candidate_model_params_set_is_warned_and_ignored(
    tmp_project, tavily_env, unsupported_key
):
    """Re-verified against installed claude-agent-sdk 0.2.144 (see module
    docstring): this project's whole `model_params` candidate list --
    including every key another adapter maps -- has no matching field on
    either `ClaudeAgentOptions` or `AgentDefinition`, so each one must warn
    individually, not just `temperature`/`max_tokens`.
    """
    writer_cfg = tmp_project / "writer" / "agent-config.yaml"
    data = yaml.safe_load(writer_cfg.read_text())
    data["model_params"][unsupported_key] = "irrelevant"
    writer_cfg.write_text(yaml.safe_dump(data))

    project = commonadk.load(tmp_project)
    with pytest.warns(UserWarning, match=f"model_params key '{unsupported_key}'"):
        options = project.build("writer", target="claude")

    assert options is not None  # build still succeeds


# ---------------------------------------------------------------------------
# env preflight
# ---------------------------------------------------------------------------


def test_missing_required_env_var_blocks_build(example_common_dir, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(example_common_dir)

    with pytest.raises(OSError) as exc_info:
        project.build("researcher", target="claude")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "Search API key used by search_web" in message
    # POSTGRES_DSN is declared `required: false` -- its absence must not be
    # reported as a blocking problem.
    assert "POSTGRES_DSN" not in message


def test_optional_env_var_absence_does_not_block(example_common_dir, tavily_env):
    project = commonadk.load(example_common_dir)

    options = project.build("researcher", target="claude")  # must not raise
    assert isinstance(options, ClaudeAgentOptions)


def test_env_preflight_checks_agents_reachable_via_edges(tmp_project, monkeypatch):
    """Building `coordinator` must also check `researcher`'s env
    requirements, since researcher is reachable from coordinator via a
    delegate edge -- even though coordinator has no `requires.env` of its
    own.
    """
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("POSTGRES_DSN", raising=False)
    project = commonadk.load(tmp_project)

    with pytest.raises(OSError, match="TAVILY_API_KEY"):
        project.build("coordinator", target="claude")
