"""
Tests for PremiseArchitect and Premise Studio API enhancements.
"""

import json
import pytest
from unittest.mock import MagicMock, patch

from models.base import LLMResponse, LLMInterface
from pipeline.premise_architect import (
    PremiseArchitect,
    clean_reasoning_and_fences,
    normalize_beat_text,
    is_prose_draft,
)
from pipeline.orchestrator import PipelineOrchestrator


class DummyLLM(LLMInterface):
    def __init__(self, response_text: str):
        self.response_text = response_text
        self.last_prompt = ""
        self.last_system = ""

    def generate(self, prompt: str, system: str = "", **kwargs) -> LLMResponse:
        self.last_prompt = prompt
        self.last_system = system
        return LLMResponse(content=self.response_text, model="dummy", provider="dummy")

    def get_name(self) -> str:
        return "dummy"

    def is_available(self) -> bool:
        return True


def test_clean_reasoning_and_fences():
    text_with_think = "<think>\nThinking about the plot...\nKey characters are Bob and Alice.\n</think>\n1. Bob enters the tavern.\n2. Alice reveals the map."
    cleaned = clean_reasoning_and_fences(text_with_think)
    assert "<think>" not in cleaned
    assert "Thinking about" not in cleaned
    assert "Bob enters the tavern." in cleaned

    text_with_fences = "```json\n{\"steps\": [\"Beat 1\"]}\n```"
    cleaned_fences = clean_reasoning_and_fences(text_with_fences)
    assert "```" not in cleaned_fences
    assert '{"steps": ["Beat 1"]}' in cleaned_fences


def test_normalize_beat_text():
    assert normalize_beat_text("1. Sarah finds the key.") == "Sarah finds the key."
    assert normalize_beat_text("Step 1: Sarah finds the key.") == "Sarah finds the key."
    assert normalize_beat_text("Beat 02 - The guards corner Sarah.") == "The guards corner Sarah."
    assert normalize_beat_text("**Beat 3:** Sarah leaps across the chasm.") == "Sarah leaps across the chasm."
    assert normalize_beat_text("- Scene 4. Escape into the catacombs.") == "Escape into the catacombs."
    assert normalize_beat_text("• Just a normal bullet beat.") == "Just a normal bullet beat."


def test_is_prose_draft():
    short_notes = "- Detective finds body\n- Suspect runs\n- Chase scene"
    assert not is_prose_draft(short_notes)

    half_story = (
        'The rain lashed against the cracked windowpane of Inspector Miller\'s damp flat. '
        '"We don\'t have much time," Theresa whispered, sliding the bloodstained dossier across the mahogany desk. '
        'Miller stared at the wax seal—the symbol of the Iron Guild. He picked up his revolver, checked the cylinder, '
        'and adjusted his heavy overcoat. "If the magistrate knows we have this, neither of us makes it past midnight." '
        'They descended into the cobblestone alley, where the shadow of a steam carriage hissed in the dark fog. '
        'Suddenly, footsteps echoed behind them. Miller turned, his weapon raised, but the figure in the doorway had vanished. '
        'A single brass cog lay on the wet stone where the watcher had stood. '
        'Theresa gasped, recognizing the mark. "It\'s him," she said, her voice trembling. "The clockmaker has found us." '
        'They ducked through the cellar grate into the sewer tunnels as whistling alarms pierced the city night.'
    )
    assert is_prose_draft(half_story)


def test_premise_orchestrator_steps_sanitization():
    raw_premise = (
        "<think>Let me outline 3 steps</think>\n"
        "```markdown\n"
        "Act 1: The Gathering\n"
        "Step 1: Kael arrives at the docklands.\n"
        "2. Vane ambushes Kael near the lighthouse.\n"
        "**Beat 3:** Kael unlocks the forgotten vault.\n"
        "```"
    )
    steps = PipelineOrchestrator._premise_steps(raw_premise)
    assert len(steps) == 3
    assert steps[0] == "Kael arrives at the docklands."
    assert steps[1] == "Vane ambushes Kael near the lighthouse."
    assert steps[2] == "Kael unlocks the forgotten vault."


def test_premise_architect_generate_continue_mode():
    json_response = {
        "summary": "Miller and Theresa race against the Iron Guild to expose the Clockmaker.",
        "setting": "Victorian London, 1888",
        "themes": ["paranoia", "conspiracy", "betrayal"],
        "story_analysis": {
            "detected_type": "half_made_story_draft",
            "cutoff_point": "Characters ducked into sewer tunnels under alarm",
            "central_conflict": "Iron Guild seeking to silence witnesses to the clockmaker conspiracy",
            "unresolved_tensions": ["Identity of the clockmaker", "Magistrate complicity"],
        },
        "characters": {
            "Inspector Miller": {
                "role": "Protagonist",
                "description": "Weary detective with a troubled past",
                "traits": ["observant", "cynical", "loyal"],
            },
            "Theresa": {
                "role": "Ally",
                "description": "Whistleblower carrying the dossier",
                "traits": ["brave", "resourceful"],
            },
        },
        "steps": [
            "Theresa brings the stolen Iron Guild dossier to Miller's flat, warning him of the impending threat.",
            "Miller and Theresa flee into the cobblestone alley as the clockmaker's watchers track their movements.",
            "They escape into the underground sewers as city alarms echo through the fog.",
            "Navigating the subterranean conduits, Miller uncovers an abandoned workshop tied to the Guild.",
            "Theresa deciphers the dossier's hidden cipher, revealing the magistrate's signature.",
            "Miller confronts the clockmaker in the clock tower, disarming the mechanism before the bells ring.",
        ],
    }

    llm = DummyLLM(json.dumps(json_response))
    result = PremiseArchitect.generate(
        idea_text="Miller and Theresa fled into the sewers...",
        mode="continue",
        target_beats=6,
        llm=llm,
    )

    assert result["status"] == "ok"
    assert len(result["steps"]) == 6
    assert "Inspector Miller" in result["characters"]
    assert result["setting"] == "Victorian London, 1888"
    assert "paranoia" in result["themes"]
    assert result["story_analysis"]["cutoff_point"] == "Characters ducked into sewer tunnels under alarm"
    assert "CONTINUE & COMPLETE HALF-MADE STORY" in llm.last_prompt


def test_premise_architect_generate_rearchitect_mode():
    json_response = {
        "summary": "A full narrative arc reimagined from the core premise.",
        "setting": "Neo-Tokyo, 2099",
        "themes": ["cybernetics", "identity"],
        "characters": {
            "Ren": {"description": "Netrunner", "traits": ["fast", "reckless"]}
        },
        "steps": [
            "Ren hacks a corporate mainframe, discovering a sentient AI fragment.",
            "Corporate enforcers raid Ren's apartment, forcing him into the neon slums.",
            "Ren seeks sanctuary with an underground rebel faction in Old Shinjuku.",
            "The AI communicates directly with Ren, revealing its creators' secret purge plan.",
            "Ren infiltrates the orbital relay tower to broadcast the AI's evidence to the city.",
            "Ren confronts the corporate director, releasing the AI into the global net to restore balance.",
        ],
    }

    llm = DummyLLM(json.dumps(json_response))
    result = PremiseArchitect.generate(
        idea_text="Ren hacks corporate files...",
        mode="rearchitect",
        target_beats=6,
        llm=llm,
    )

    assert result["status"] == "ok"
    assert len(result["steps"]) == 6
    assert "Ren" in result["characters"]
    assert "RE-ARCHITECT FULL STORY FROM BEGINNING" in llm.last_prompt


def test_premise_architect_fallback_plain_text():
    plain_text_llm_output = (
        "Here are the narrative steps:\n"
        "1. Alex uncovers the encrypted relic in the ruins.\n"
        "2. The rival syndicate ambushes the expedition party.\n"
        "3. Alex decodes the final glyph, securing the sanctuary before the collapse.\n"
    )
    llm = DummyLLM(plain_text_llm_output)
    result = PremiseArchitect.generate(
        idea_text="Alex finds a relic...",
        mode="auto",
        llm=llm,
    )

    assert result["status"] == "ok"
    assert len(result["steps"]) == 3
    assert result["steps"][0] == "Alex uncovers the encrypted relic in the ruins."
    assert result["steps"][1] == "The rival syndicate ambushes the expedition party."


def test_premise_architect_refine():
    refined_json = {
        "steps": [
            "Beat 1: Kael enters the forbidden sanctum.",
            "Beat 2: The guardian tests Kael's resolve.",
            "Beat 3: Kael retrieves the heartstone, sealing the rift.",
        ]
    }
    llm = DummyLLM(json.dumps(refined_json))
    refined = PremiseArchitect.refine(
        steps=["Kael enters", "The guardian tests", "Kael retrieves"],
        llm=llm,
    )
    assert len(refined) == 3
    assert refined[0] == "Kael enters the forbidden sanctum."


def test_premise_architect_expand():
    expanded_json = {
        "steps": [
            "Beat 1: Kael enters the sanctum.",
            "Beat 2: Kael discovers ancient markings revealing the guardian's weakness.",
            "Beat 3: The guardian awakens and blocks the threshold.",
            "Beat 4: Kael uses the markings to outwit the guardian.",
            "Beat 5: Kael retrieves the heartstone and escapes.",
        ]
    }
    llm = DummyLLM(json.dumps(expanded_json))
    expanded = PremiseArchitect.expand(
        steps=["Beat 1", "Beat 2", "Beat 3"],
        llm=llm,
    )
    assert len(expanded) == 5


def test_premise_api_routes(tmp_path, monkeypatch):
    import app as server
    # Set up dummy project in test directory
    project_dir = tmp_path / "projects" / "test_premise_proj"
    project_dir.mkdir(parents=True)
    state = {
        "metadata": {
            "title": "Test Story",
            "genre": "fantasy",
            "premise": "1. Old beat",
            "setting": "Ancient Rome",
            "themes": ["honor"],
        },
        "characters": {
            "Marcus": {"description": "Centurion", "traits": ["loyal"]}
        },
        "world": {},
        "plot": {},
    }
    (project_dir / "state.json").write_text(json.dumps(state))

    monkeypatch.setattr(server.config, "PROJECTS_DIR", str(tmp_path / "projects"))

    mock_llm = DummyLLM(
        json.dumps({
            "summary": "Marcus uncovers corruption.",
            "setting": "Rome 44 BC",
            "themes": ["duty", "rebellion"],
            "characters": {
                "Marcus": {"description": "Soldier", "traits": ["fierce"]},
                "Senator Cassius": {"description": "Conspirator", "traits": ["cunning"]}
            },
            "steps": [
                "1. Marcus discovers the hidden ledger.",
                "2. Cassius attempts to bribe Marcus.",
                "3. Marcus presents the evidence to the Senate."
            ],
            "story_analysis": {
                "mode_applied": "continue",
                "cutoff_point": "Ledger discovered"
            }
        })
    )
    monkeypatch.setattr(server, "_get_llm_for_premise", lambda req_model: mock_llm)

    client = server.create_app().test_client()

    # Test Generate API
    resp = client.post(
        "/api/project/test_premise_proj/premise/generate",
        json={
            "idea_text": "Marcus found a hidden ledger in the temple...",
            "mode": "continue",
            "target_chapters": 5,
            "apply_to_bible": True,
        }
    )
    assert resp.status_code == 200
    data = resp.json
    assert data["status"] == "ok"
    assert len(data["steps"]) == 3
    assert "Senator Cassius" in data["characters"]

    # Verify state was updated with newly discovered characters & setting
    with open(project_dir / "state.json", "r") as f:
        saved_state = json.load(f)
    assert "Senator Cassius" in saved_state["characters"]
    assert saved_state["metadata"]["setting"] == "Rome 44 BC"

    # Test Refine API
    resp_refine = client.post(
        "/api/project/test_premise_proj/premise/refine",
        json={"steps": data["steps"]}
    )
    assert resp_refine.status_code == 200
    assert "steps" in resp_refine.json

    # Test Expand API
    resp_expand = client.post(
        "/api/project/test_premise_proj/premise/expand",
        json={"steps": data["steps"], "target_beats": 6}
    )
    assert resp_expand.status_code == 200
    assert "steps" in resp_expand.json


def test_premise_standalone_api_route(tmp_path, monkeypatch):
    import app as server
    monkeypatch.setattr(server.config, "PROJECTS_DIR", str(tmp_path / "projects"))

    mock_llm = DummyLLM(
        json.dumps({
            "title": "Echoes of the Obsidian Spire",
            "genre": "Dark Fantasy",
            "summary": "An exile returns to shatter an obsidian empire.",
            "setting": "The Wastes of Kar-Drak",
            "themes": ["vengeance", "legacy"],
            "characters": {
                "Vael": {"description": "Exiled shadow-blade", "traits": ["relentless", "guilt-ridden"]},
                "High Inquisitor Mor": {"description": "Ruler of the Spire", "traits": ["zealous", "iron-willed"]}
            },
            "steps": [
                "1. Vael returns to Kar-Drak under the crimson eclipse.",
                "2. Mor's sentinels ambush Vael at the Black Gate.",
                "3. Vael breaches the inner sanctum to face Mor."
            ],
            "story_analysis": {
                "detected_type": "half_made_story_draft",
                "mode_applied": "continue",
                "cutoff_point": "Ambush at Black Gate",
                "central_conflict": "Vael versus Mor's theocratic order"
            }
        })
    )
    monkeypatch.setattr(server, "_get_llm_for_premise", lambda req_model: mock_llm)

    client = server.create_app().test_client()

    resp = client.post(
        "/api/premise/generate",
        json={
            "idea_text": "Vael walked into the wastes of Kar-Drak...",
            "mode": "continue",
            "target_chapters": 5,
        }
    )
    assert resp.status_code == 200
    data = resp.json
    assert data["status"] == "ok"
    assert data["title"] == "Echoes of the Obsidian Spire"
    assert data["genre"] == "Dark Fantasy"
    assert len(data["steps"]) == 3
    assert "Vael" in data["characters"]
    assert data["characters"]["Vael"]["traits"] == ["relentless", "guilt-ridden"]
    assert data["story_analysis"]["mode_applied"] == "continue"

