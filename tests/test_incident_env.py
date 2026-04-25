import json
import unittest

from generated_envs.incident_triage_env import IncidentTriageEnv


def action(params, confidence=0.8):
    return (
        "<reasoning>\nThe evidence supports this diagnosis.\n</reasoning>\n"
        "<action>\n"
        + json.dumps(
            {
                "action_type": "diagnose_incident",
                "params": params,
                "confidence": confidence,
            }
        )
        + "\n</action>"
    )


class IncidentTriageEnvTests(unittest.TestCase):
    def test_reset_step_state_across_eci(self):
        for eci in range(1, 6):
            env = IncidentTriageEnv(eci=eci, max_steps=20)
            obs = env.reset()
            self.assertEqual(obs["context"]["eci"], eci)
            truth = env.state()["task"]["ground_truth"]
            params = {
                **truth,
                "evidence": "logs and metrics point to the matching root cause",
            }
            next_obs, reward, done, info = env.step(action(params))
            self.assertIsInstance(next_obs, dict)
            self.assertIsInstance(reward, float)
            self.assertGreaterEqual(reward, 0.0)
            self.assertLessEqual(reward, 1.0)
            self.assertIn("solve_rate_signal", info)
            self.assertEqual(info["eci"], eci)
            self.assertIn("reward_breakdown", info)

    def test_malformed_action_penalized(self):
        env = IncidentTriageEnv(eci=3, max_steps=20)
        env.reset()
        _, reward, _, info = env.step("not json")
        self.assertEqual(reward, 0.0)
        self.assertLess(info["reward_breakdown"]["penalty"], 0.0)

    def test_ground_truth_not_in_observation(self):
        env = IncidentTriageEnv(eci=1, max_steps=20)
        obs = env.reset()
        self.assertNotIn("ground_truth", json.dumps(obs))


if __name__ == "__main__":
    unittest.main()

