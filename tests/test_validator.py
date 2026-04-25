import unittest
from pathlib import Path

from src.validator.validate_env import validate_env_file


class ValidatorTests(unittest.TestCase):
    def test_demo_env_validates(self):
        result = validate_env_file(
            Path("generated_envs") / "incident_triage_env.py",
            class_name="IncidentTriageEnv",
        )
        self.assertTrue(result.ok, result.as_dict())


if __name__ == "__main__":
    unittest.main()

