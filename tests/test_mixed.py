"""Tests for mixed-target spawning (`Project.build_mixed`, `mixed.py`).

See `docs/mixed-target-design.md` for the full design this exercises: the
three-layer model (per-agent build -> per-runtime unit/"island" ->
coordinator), the island algorithm, and the verified supported/unsupported
cross-runtime source targets.

Individual SDKs are gated with `pytest.importorskip` *inside* each test
body (matching `tests/test_hypothesis.py`'s style, not a whole-module
`importorskip`), since different tests here need different subsets of the
six SDKs -- this file must still collect and run its SDK-independent tests
(the island/error-path ones) with zero SDKs installed.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import yaml

import commonadk

REPO_ROOT = Path(__file__).resolve().parents[1]
MIXED_CREW_COMMON = REPO_ROOT / "examples" / "mixed-crew" / "common"

_IMPORT_MODULE = {
    "google-adk": "google.adk",
    "openai": "agents",
    "claude": "claude_agent_sdk",
    "crewai": "crewai",
    "autogen": "autogen_agentchat",
    "langgraph": "langgraph",
}


def _skip_unless_installed(*targets: str) -> None:
    for target in targets:
        pytest.importorskip(_IMPORT_MODULE[target])


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text())


def _dump_yaml(path: Path, data) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False))


@pytest.fixture()
def tavily_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "test-key")
    monkeypatch.delenv("POSTGRES_DSN", raising=False)


@pytest.fixture()
def provider_keys_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GEMINI_API_KEY", "AIza-test")
    monkeypatch.setenv("GOOGLE_API_KEY", "AIza-test")


@pytest.fixture()
def crewai_offline_env(monkeypatch):
    # Belt-and-braces, mirroring test_adapter_crewai.py: construction never
    # touches the network either way, but keep this file independent of
    # that accident holding forever.
    monkeypatch.setenv("CREWAI_DISABLE_TELEMETRY", "true")
    monkeypatch.setenv("OTEL_SDK_DISABLED", "true")


@pytest.fixture()
def research_crew_copy(tmp_path):
    dest = tmp_path / "common"
    shutil.copytree(REPO_ROOT / "examples" / "research-crew" / "common", dest)
    return dest


@pytest.fixture()
def mixed_crew_copy(tmp_path):
    dest = tmp_path / "common"
    shutil.copytree(MIXED_CREW_COMMON, dest)
    return dest


def _set_runtime(project_dir: Path, agent: str, runtime: str, model: str | None = None) -> None:
    cfg_path = project_dir / agent / "agent-config.yaml"
    data = _load_yaml(cfg_path)
    data["runtime"] = runtime
    if model is not None:
        data["model"] = model
    _dump_yaml(cfg_path, data)


# ---------------------------------------------------------------------------
# compatibility: no agent sets `runtime:` -> build_mixed matches build()
# ---------------------------------------------------------------------------


def test_build_mixed_matches_build_when_no_runtime_set(example_common_dir, tavily_env):
    """The compatibility contract (docs/mixed-target-design.md, "What
    `runtime:` means now"): a project with no agent's `runtime:` set builds
    identically under `build_mixed` and `build` -- one island, same native
    object shape.
    """
    _skip_unless_installed("google-adk")

    project = commonadk.load(example_common_dir)

    direct = project.build("coordinator", target="google-adk")
    mixed = project.build_mixed("coordinator", default_target="google-adk")

    assert list(mixed.units.keys()) == ["coordinator"]
    assert mixed.units["coordinator"].runtime == "google-adk"
    assert mixed.units["coordinator"].members == ["coordinator", "researcher", "writer"]
    assert mixed.cross_edges == []

    native = mixed.entry_native
    assert native.name == direct.name == "coordinator"
    assert [t.__name__ for t in native.tools] == [t.__name__ for t in direct.tools]
    assert [a.name for a in native.sub_agents] == [a.name for a in direct.sub_agents]


def test_no_runtime_check_imports_sdk_when_unset(tmp_path, monkeypatch):
    """A project that sets no agent's `runtime:` never imports any adapter
    module at load time -- the one deliberate SDK-import exception this
    feature adds only applies when `runtime:` is actually set (design doc,
    "What `runtime:` means now").
    """
    import importlib

    import commonadk.adapters as adapters_pkg

    calls: list[str] = []
    real_import_module = importlib.import_module

    def spying_import_module(name, *args, **kwargs):
        calls.append(name)
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(adapters_pkg, "import_module", spying_import_module)

    dest = tmp_path / "common"
    shutil.copytree(REPO_ROOT / "examples" / "research-crew" / "common", dest)
    commonadk.load(dest)

    assert calls == []


# ---------------------------------------------------------------------------
# a real two-runtime project
# ---------------------------------------------------------------------------


def test_two_runtime_project_builds_native_types(mixed_crew_copy, crewai_offline_env):
    """examples/mixed-crew: coordinator (default target) -> specialist
    (`runtime: crewai`) builds two islands, each the right native type, and
    the cross-runtime edge lands as a callable tool on the coordinator.
    """
    _skip_unless_installed("google-adk", "crewai")
    from crewai import Crew
    from google.adk.agents import Agent as ADKAgent

    project = commonadk.load(mixed_crew_copy)
    mixed = project.build_mixed("coordinator", default_target="google-adk")

    assert mixed.agent_runtime == {"coordinator": "google-adk", "specialist": "crewai"}
    assert mixed.cross_edges == [("coordinator", "specialist")]
    assert set(mixed.units) == {"coordinator", "specialist"}

    coordinator_unit = mixed.unit_for("coordinator")
    specialist_unit = mixed.unit_for("specialist")
    assert coordinator_unit.runtime == "google-adk"
    assert isinstance(coordinator_unit.native, ADKAgent)
    assert specialist_unit.runtime == "crewai"
    assert isinstance(specialist_unit.native, Crew)

    tool_names = [getattr(t, "__name__", None) for t in coordinator_unit.native.tools]
    assert "greet" in tool_names
    assert "transfer_to_specialist" in tool_names


# ---------------------------------------------------------------------------
# island computation: same-runtime, connected agents share one native build
# ---------------------------------------------------------------------------


def test_island_shares_one_native_subgraph_for_connected_same_runtime_agents(
    research_crew_copy, tavily_env, crewai_offline_env
):
    """researcher and writer both pinned to crewai, connected by their own
    `researcher -> writer` edge: they must land in ONE runtime unit, built
    as one native `crewai.Crew` sub-graph (true native wiring), while
    coordinator (still google-adk) is a separate, single-member unit and
    the only cross-runtime edge is coordinator -> researcher.
    """
    _skip_unless_installed("google-adk", "crewai")
    from crewai import Process

    _set_runtime(research_crew_copy, "researcher", "crewai", model="openai/gpt-4o")
    _set_runtime(research_crew_copy, "writer", "crewai", model="openai/gpt-4o")

    project = commonadk.load(research_crew_copy)
    mixed = project.build_mixed("coordinator", default_target="google-adk")

    assert mixed.agent_runtime == {
        "coordinator": "google-adk",
        "researcher": "crewai",
        "writer": "crewai",
    }
    assert set(mixed.units) == {"coordinator", "researcher"}
    assert mixed.units["researcher"].members == ["researcher", "writer"]
    assert mixed.cross_edges == [("coordinator", "researcher")]

    # True native wiring within the island: one Crew, both agents members
    # of it (researcher as hierarchical manager, writer as its delegate) --
    # not two separately-built objects stitched together after the fact.
    crew = mixed.units["researcher"].native
    assert crew.process == Process.hierarchical
    assert crew.manager_agent.role == "researcher"
    assert [a.role for a in crew.agents] == ["writer"]

    coordinator_tools = [
        getattr(t, "__name__", None) for t in mixed.entry_native.tools
    ]
    assert "transfer_to_researcher" in coordinator_tools


# ---------------------------------------------------------------------------
# cross-runtime source support: verified per adapter (design doc table)
# ---------------------------------------------------------------------------


def _assert_has_transfer_tool(native, runtime: str, dst_name: str) -> None:
    tool_name = f"transfer_to_{dst_name}"
    if runtime == "google-adk":
        assert any(getattr(t, "__name__", None) == tool_name for t in native.tools)
    elif runtime == "openai":
        assert any(getattr(t, "name", None) == tool_name for t in native.tools)
    elif runtime == "crewai":
        from crewai import Process

        root_agent = native.manager_agent if native.process == Process.hierarchical else native.agents[0]
        assert any(getattr(t, "name", None) == tool_name for t in root_agent.tools)
    elif runtime == "claude":
        assert any(tool_name in name for name in native.allowed_tools)
    else:  # pragma: no cover - not a source-capable target
        raise AssertionError(f"no transfer-tool check defined for {runtime!r}")


@pytest.mark.parametrize(
    "source_runtime, dest_runtime, coordinator_model",
    [
        ("google-adk", "crewai", None),
        ("openai", "google-adk", None),
        ("claude", "google-adk", "anthropic/claude-sonnet-5"),
        ("crewai", "google-adk", None),
    ],
)
def test_cross_runtime_source_bridges_for_every_supported_target(
    mixed_crew_copy,
    provider_keys_env,
    crewai_offline_env,
    source_runtime,
    dest_runtime,
    coordinator_model,
):
    """One test per v1 source-capable target (design doc,
    "Supported/unsupported cross-runtime source targets"): the island
    builds, and the cross-runtime edge lands as a real, callable,
    correctly-named tool on the source island's root agent.
    """
    _skip_unless_installed(source_runtime, dest_runtime)

    _set_runtime(mixed_crew_copy, "coordinator", source_runtime, model=coordinator_model)
    _set_runtime(mixed_crew_copy, "specialist", dest_runtime)

    project = commonadk.load(mixed_crew_copy)
    mixed = project.build_mixed("coordinator", default_target=source_runtime)

    assert mixed.agent_runtime["coordinator"] == source_runtime
    assert mixed.agent_runtime["specialist"] == dest_runtime
    assert mixed.cross_edges == [("coordinator", "specialist")]

    _assert_has_transfer_tool(mixed.entry_native, source_runtime, "specialist")


def test_cross_runtime_source_bridge_actually_invokes_destination(
    mixed_crew_copy, crewai_offline_env, monkeypatch
):
    """End-to-end proof the bridge is a real, callable function, not just
    an attached-looking tool object: calling it runs the destination
    island's own native execution path (`Crew.kickoff()` here) and returns
    its text result -- no network involved, since both agents' tools are
    pure Python and the LLM call itself is monkeypatched out.
    """
    _skip_unless_installed("google-adk", "crewai")
    from crewai import Crew

    project = commonadk.load(mixed_crew_copy)
    mixed = project.build_mixed("coordinator", default_target="google-adk")

    coordinator = mixed.entry_native
    transfer = next(t for t in coordinator.tools if getattr(t, "__name__", None) == "transfer_to_specialist")

    class _FakeResult:
        raw = "the specialist crew's canned answer"

    captured: dict = {}

    def fake_kickoff(self):
        captured["description"] = self.tasks[0].description
        return _FakeResult()

    monkeypatch.setattr(Crew, "kickoff", fake_kickoff)

    result = transfer("please handle this")
    assert result == "the specialist crew's canned answer"
    assert captured["description"] == "please handle this"


# ---------------------------------------------------------------------------
# unsupported cross-runtime source targets
# ---------------------------------------------------------------------------


def test_unsupported_cross_runtime_source_errors(research_crew_copy, tavily_env, provider_keys_env):
    """researcher (`langgraph`, unsupported as a cross-runtime source) ->
    writer (`google-adk`): a clear error naming the agent, its runtime, and
    why -- never a silent no-op. See design doc's evidence table.
    """
    _skip_unless_installed("google-adk", "langgraph")

    _set_runtime(research_crew_copy, "researcher", "langgraph")

    project = commonadk.load(research_crew_copy)

    with pytest.raises(ValueError) as exc_info:
        project.build_mixed("coordinator", default_target="google-adk")

    message = str(exc_info.value)
    assert "researcher" in message
    assert "langgraph" in message
    assert "not one of the cross-runtime source-capable targets" in message


def test_cross_runtime_edge_must_source_at_island_root(research_crew_copy, tavily_env):
    """coordinator -> researcher -> writer, with only `writer` pinned to a
    different runtime: the cross-runtime edge (researcher -> writer)
    originates at `researcher`, which is NOT its island's root
    (`coordinator` is, since coordinator+researcher share google-adk and
    coordinator reaches researcher) -- v1 requires root-sourced
    cross-runtime edges (design doc, "Cross-runtime edges").
    """
    _skip_unless_installed("google-adk", "crewai")

    _set_runtime(research_crew_copy, "writer", "crewai")

    project = commonadk.load(research_crew_copy)

    with pytest.raises(ValueError) as exc_info:
        project.build_mixed("coordinator", default_target="google-adk")

    message = str(exc_info.value)
    assert "researcher" in message
    assert "writer" in message
    assert "not the root of its runtime unit" in message
    assert "coordinator" in message


def _minimal_agent_files(common: Path, name: str, model: str = "fast", runtime: str | None = None) -> None:
    (common / name).mkdir(parents=True, exist_ok=True)
    runtime_line = f"runtime: {runtime}\n" if runtime else ""
    (common / name / "agent-config.yaml").write_text(
        f"name: {name}\ndescription: agent {name}\nmodel: {model}\n"
        f"tools: []\nrequires:\n  env: []\n{runtime_line}"
    )
    (common / name / "skill.md").write_text(f"# {name}\n")
    (common / name / "tools.py").write_text("")


def test_crewai_manager_cannot_be_cross_runtime_source(tmp_path, crewai_offline_env, provider_keys_env):
    """coordinator (crewai) delegates to helper (crewai, same island --
    coordinator becomes the island's hierarchical manager) AND has a
    cross-runtime edge to specialist (google-adk): CrewAI managers cannot
    hold tools, so this is a documented, specific error -- not a silent
    tool-drop.
    """
    _skip_unless_installed("google-adk", "crewai")

    common = tmp_path / "common"
    common.mkdir()
    (common / "config.yaml").write_text(
        "name: crewai-manager-source\nentry: coordinator\n"
        "targets: [google-adk, crewai]\ndefault_model: fast\n"
        "model_aliases:\n  fast: openai/gpt-4o\n"
    )
    (common / "interactions.yaml").write_text(
        "entry: coordinator\nedges:\n"
        "  - from: coordinator\n    to: helper\n    type: delegate\n"
        "  - from: coordinator\n    to: specialist\n    type: delegate\n"
    )
    _minimal_agent_files(common, "coordinator", runtime="crewai")
    _minimal_agent_files(common, "helper", runtime="crewai")
    _minimal_agent_files(common, "specialist", runtime="google-adk")

    project = commonadk.load(common)

    with pytest.raises(ValueError) as exc_info:
        project.build_mixed("coordinator", default_target="google-adk")

    message = str(exc_info.value)
    assert "coordinator" in message
    assert "manager" in message.lower()


# ---------------------------------------------------------------------------
# island computation: a multi-root island is rejected, not silently partial
# ---------------------------------------------------------------------------


def test_multi_root_island_errors():
    """Three agents in one island, `x -> y` and `z -> y`: nothing reaches
    everything (`x` can't reach `z`, `z` can't reach `x`, `y` reaches
    neither) -- no single root can build this island, so `_pick_root`
    raises rather than silently building only part of it (design doc,
    "Island computation"). Exercises the algorithm directly against plain
    `InteractionEdge`s -- no full project/adapter needed for this check.
    """
    from commonadk.mixed import _pick_root
    from commonadk.models import InteractionEdge

    edges = [
        InteractionEdge(from_="x", to="y", type="delegate"),
        InteractionEdge(from_="z", to="y", type="delegate"),
    ]

    with pytest.raises(ValueError) as exc_info:
        _pick_root({"x", "y", "z"}, edges, preferred=None)

    message = str(exc_info.value)
    for name in ("x", "y", "z"):
        assert name in message


def test_pick_root_prefers_entry_agent_when_it_qualifies():
    """A qualifying non-preferred candidate must not be picked over the
    overall build entry when the entry itself also qualifies -- the
    compatibility-preserving choice (design doc, "Island computation").
    """
    from commonadk.mixed import _pick_root
    from commonadk.models import InteractionEdge

    edges = [
        InteractionEdge(from_="coordinator", to="researcher", type="delegate"),
        InteractionEdge(from_="researcher", to="writer", type="handoff"),
    ]

    root = _pick_root({"coordinator", "researcher", "writer"}, edges, preferred="coordinator")
    assert root == "coordinator"


# ---------------------------------------------------------------------------
# env preflight spans every runtime, not just one
# ---------------------------------------------------------------------------


def test_env_preflight_spans_runtimes(research_crew_copy, crewai_offline_env):
    """researcher (pinned to crewai, requires TAVILY_API_KEY) and writer
    (pinned to openai, given a synthetic required env var here) both miss
    their required env var: one OSError names both agents and both
    runtimes, raised before any island is built.
    """
    _skip_unless_installed("crewai", "openai")

    _set_runtime(research_crew_copy, "researcher", "crewai", model="openai/gpt-4o")
    _set_runtime(research_crew_copy, "writer", "openai")

    writer_cfg = research_crew_copy / "writer" / "agent-config.yaml"
    data = _load_yaml(writer_cfg)
    data["requires"] = {
        "env": [{"name": "WRITER_SECRET", "description": "test-only", "required": True}]
    }
    _dump_yaml(writer_cfg, data)

    project = commonadk.load(research_crew_copy)

    with pytest.raises(OSError) as exc_info:
        project.build_mixed("coordinator", default_target="google-adk")

    message = str(exc_info.value)
    assert "TAVILY_API_KEY" in message
    assert "researcher" in message and "crewai" in message
    assert "WRITER_SECRET" in message
    assert "writer" in message and "openai" in message
