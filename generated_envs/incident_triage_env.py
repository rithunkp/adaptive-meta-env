"""
Incident triage environment where agents must diagnose root causes of system issues.
Models causal relationships between infrastructure problems and their symptoms.
Rewards efficient diagnosis and recovery while penalizing incorrect actions.
"""

import random
from typing import Dict, Tuple
from dataclasses import dataclass

@dataclass
class SystemState:
    cpu_utilization_high: bool
    memory_leak_detected: bool
    network_packet_loss: bool
    runbook_available: bool
    diagnosis: str = ""
    recovery_time: float = 0.0
    steps_taken: int = 0
    max_steps: int = 10

class IncidentTriageEnv:
    """Environment for diagnosing infrastructure incidents through causal reasoning."""
    
    def __init__(self, eci: int = 1, max_steps: int = 10):
        self.eci = eci
        self._state = SystemState(
            cpu_utilization_high=False,
            memory_leak_detected=False,
            network_packet_loss=False,
            runbook_available=random.random() > 0.5 if eci >= 2 else True,
            max_steps=max_steps
        )
        self.possible_issues = [
            "payment_gateway_timeout",
            "cache_replication_lag",
            "queue_consumer_crash_loop"
        ]
        self.valid_actions = self.possible_issues + ["check_symptoms", "consult_runbook", "no_action"]
        self.current_scenario = None
        self._generate_incident()
        
    def _generate_incident(self) -> None:
        """Randomly generate a root cause based on ECI level."""
        scenarios = [
            {
                "service": "checkout",
                "alert": "Elevated checkout latency and payment failures",
                "root_cause": "payment_gateway_timeout",
                "logs": ["payment gateway timeout", "retry budget exhausted"],
                "runbook": "Check dependency health before scaling."
            },
            {
                "service": "inventory",
                "alert": "Product availability reads are stale",
                "root_cause": "cache_replication_lag",
                "logs": ["cache replica behind primary", "memory usage high"],
                "runbook": "Compare cache freshness before restarting workers."
            },
            {
                "service": "orders",
                "alert": "Order processing queue depth increasing",
                "root_cause": "queue_consumer_crash_loop",
                "logs": ["consumer CPU at 95%", "processing lag"],
                "runbook": "Check consumer worker health and scaling."
            }
        ]
        self.current_scenario = random.choice(scenarios)
        root_cause = self.current_scenario["root_cause"]
        
        # Store root_cause for reward calculation
        self._root_cause = root_cause
        
        # Add decoys at higher ECI
        if self.eci >= 3:
            decoys = [c for c in self.possible_issues if c != root_cause]
            for decoy in random.sample(decoys, k=min(self.eci-2, len(decoys))):
                # Randomly add false positives (decoys)
                pass  # decoys are handled in action parsing
        
        if self.eci >= 4:
            self._state.runbook_available = random.random() > 0.6
    
    def _get_symptoms(self) -> Dict[str, bool]:
        """Generate observable symptoms based on root causes."""
        symptoms = {
            "checkout_service_degraded": False,
            "database_latency_high": False,
            "api_error_rate_increased": False,
            "payment_gateway_timeout": False
        }
        
        if self._state.cpu_utilization_high:
            symptoms["database_latency_high"] = True
            symptoms["checkout_service_degraded"] = True
            
        if self._state.memory_leak_detected:
            symptoms["api_error_rate_increased"] = True
            symptoms["checkout_service_degraded"] = True
            
        if self._state.network_packet_loss:
            symptoms["payment_gateway_timeout"] = True
            symptoms["checkout_service_degraded"] = True
            
        if self.eci >= 2:
            # Add noise to symptoms
            for symptom in symptoms:
                if random.random() < 0.1 * self.eci and not symptoms[symptom]:
                    symptoms[symptom] = True
        
        return symptoms
    
    def _calculate_recovery_time(self) -> None:
        """Calculate recovery time based on diagnosis accuracy and runbook availability."""
        base_time = 1.0
        if self._state.diagnosis:
            if self._state.runbook_available:
                base_time *= 0.5
            if self.eci >= 5:
                base_time *= max(0.5, self._state.steps_taken / self._state.max_steps)
            self._state.recovery_time = base_time
    
    def reset(self) -> Dict:
        """Reset the environment with a new random incident."""
        self.__init__(eci=self.eci)
        symptoms = self._get_symptoms()
        return {
            "task": "Diagnose the root cause of system issues",
            "context": {
                "service": self.current_scenario["service"],
                "alert": self.current_scenario["alert"],
                "logs": self.current_scenario["logs"],
                "runbook": self.current_scenario["runbook"] if self._state.runbook_available else "",
                "symptoms": symptoms,
                "runbook_available": self._state.runbook_available,
                "eci": self.eci
            },
            "step": self._state.steps_taken
        }
    
    def step(self, action: str) -> Tuple[Dict, float, bool, Dict]:
        """Process an agent action and return new observation, reward, done, and info."""
        self._state.steps_taken += 1
        
        # Parse the formatted action string if it contains JSON
        parsed_diagnosis = None
        breakdown = {"correctness": 0.0, "efficiency": 0.0, "quality": 0.2, "penalty": 0.0}
        
        if "<action>" in action and "</action>" in action:
            try:
                import json
                start = action.index("<action>") + 8
                end = action.index("</action>")
                action_json = json.loads(action[start:end].strip())
                
                # Extract diagnosis from params
                if action_json.get("action_type") == "diagnose_incident":
                    params = action_json.get("params", {})
                    parsed_diagnosis = params.get("root_cause", "")
            except (json.JSONDecodeError, ValueError, KeyError):
                parsed_diagnosis = None
        
        # Calculate reward components
        correctness = 0.0
        efficiency = 0.0
        quality = 0.2
        penalty = 0.0
        
        # Correctness: Did the diagnosis match the true root cause?
        if parsed_diagnosis == self._root_cause:
            correctness = 0.5
            self._state.diagnosis = parsed_diagnosis
            self._calculate_recovery_time()
        elif parsed_diagnosis and parsed_diagnosis in self.possible_issues:
            correctness = -0.3
            penalty = -0.1 * self._state.steps_taken / self._state.max_steps
        
        # Efficiency: Bonus if solved quickly, penalty if took too long
        if self._state.diagnosis:
            efficiency = 0.1 if self._state.steps_taken <= 3 else max(0.0, 0.1 - 0.02 * self._state.steps_taken)
        
        # Recovery time bonus
        total_reward = 0.0
        if self._state.diagnosis:
            recovery_bonus = 0.2 * (1 - min(self._state.recovery_time, 1.0))
            total_reward = min(1.0, max(0.0, correctness + efficiency + quality + penalty + recovery_bonus))
        else:
            total_reward = min(1.0, max(0.0, correctness + efficiency + quality + penalty))
        
        breakdown = {
            "correctness": round(correctness, 4),
            "efficiency": round(efficiency, 4),
            "quality": round(quality, 4),
            "penalty": round(penalty, 4)
        }
        
        done = (self._state.diagnosis != "") or (self._state.steps_taken >= self._state.max_steps)
        
        obs = {
            "task": "Diagnose the root cause of system issues",
            "context": {
                "service": self.current_scenario["service"],
                "alert": self.current_scenario["alert"],
                "logs": self.current_scenario["logs"],
                "runbook": self.current_scenario["runbook"] if self._state.runbook_available else "",
                "runbook_available": self._state.runbook_available,
                "diagnosis_made": self._state.diagnosis != "",
                "steps_remaining": self._state.max_steps - self._state.steps_taken
            },
            "step": self._state.steps_taken
        }
        
        info = {
            "solve_rate_signal": total_reward,
            "eci": self.eci,
            "internal_state": self._state.__dict__,
            "reward_breakdown": breakdown
        }
        
        return obs, total_reward, done, info
    
    def state(self) -> Dict:
        """Return full internal state for debugging."""
        return {
            "system_state": self._state.__dict__,
            "possible_issues": self.possible_issues,
            "valid_actions": self.valid_actions
        }