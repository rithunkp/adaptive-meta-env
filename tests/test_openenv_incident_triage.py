import json
import sys
import unittest
from pathlib import Path


ENV_ROOT = Path(__file__).resolve().parents[1] / "envs" / "incident_triage"
sys.path.insert(0, str(ENV_ROOT))

try:
    from models import IncidentTriageAction
    from server.incident_triage_environment import IncidentTriageEnvironment
except ModuleNotFoundError as exc:
    if exc.name and exc.name.startswith("openenv"):
        IncidentTriageAction = None
        IncidentTriageEnvironment = None
    else:
        raise


@unittest.skipIf(IncidentTriageEnvironment is None, "openenv-core is not installed")
class OpenEnvIncidentTriageTests(unittest.TestCase):
    def test_reset_hides_ground_truth(self):
        env = IncidentTriageEnvironment(eci=3, max_steps=20)
        obs = env.reset(seed=7)
        self.assertNotIn("ground_truth", json.dumps(obs.model_dump()))
        self.assertEqual(obs.context["eci"], 3)

    def test_valid_action_returns_reward_metadata(self):
        env = IncidentTriageEnvironment(eci=1, max_steps=20)
        env.reset(seed=1)
        truth = env.state.task["ground_truth"]
        obs = env.step(
            IncidentTriageAction(
                action_type="diagnose_incident",
                params={
                    **truth,
                    "evidence": "logs and metrics point to the matching root cause",
                },
                confidence=0.8,
                reasoning="The observed signals support this diagnosis.",
            )
        )
        self.assertIsInstance(obs.reward, float)
        self.assertIn("reward_breakdown", obs.metadata)
        self.assertIn("solve_rate_signal", obs.metadata)

    def test_malformed_raw_response_penalized(self):
        env = IncidentTriageEnvironment(eci=3, max_steps=20)
        env.reset(seed=2)
        obs = env.step(IncidentTriageAction(raw_response="not json"))
        self.assertEqual(obs.reward, 0.0)
        self.assertLess(obs.metadata["reward_breakdown"]["penalty"], 0.0)

    def test_string_params_are_parsed(self):
        env = IncidentTriageEnvironment(eci=1, max_steps=20)
        env.reset(seed=3)
        truth = env.state.task["ground_truth"]
        payload = json.dumps(
            {
                **truth,
                "evidence": "payment gateway timeout in logs",
            }
        )
        obs = env.step(
            IncidentTriageAction(
                action_type="diagnose_incident",
                params=payload,
                confidence=0.8,
                reasoning="Signals align with dependency timeout.",
            )
        )
        self.assertGreaterEqual(obs.reward, 0.0)
        self.assertIsInstance(obs.metadata["reward_breakdown"], dict)


if __name__ == "__main__":
    unittest.main()
