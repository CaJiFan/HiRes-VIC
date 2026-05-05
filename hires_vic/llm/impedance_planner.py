"""
LLM/VLM Impedance Planner for contact-rich manipulation.
Given task state, returns a stiffness prior (log-space SPD hint)
that biases the RL policy's action distribution.

Backend options (set via `backend` arg):
  "openai"  -- OpenAI API (GPT-4o, GPT-4o-mini, etc.)  needs OPENAI_API_KEY
  "ollama"  -- Local Ollama server (llama3.2, mistral, etc.) no key needed
               Install: https://ollama.com  then `ollama pull llama3.2`

Examples:
  LLMImpedancePlanner(backend="openai", model="gpt-4o-mini")
  LLMImpedancePlanner(backend="ollama", model="llama3.2")
"""
import numpy as np
from dataclasses import dataclass
from openai import OpenAI

# In log-space Mandel parameterization matching spd_grl_map input:
# [m11, m22, m33, m12, m13, m23] (diagonal-dominant = anisotropic stiffness)
IMPEDANCE_MODES = {
    "approach": {
        "log_kp": np.array([3.5, 3.5, 3.0, 0.0, 0.0, 0.0]),   # isotropic, medium
        "kp_rot": np.array([15.0, 15.0, 15.0]),
        "description": "Pre-contact approach: uniform medium stiffness",
    },
    "contact_edge": {
        "log_kp": np.array([4.5, 4.5, 2.5, 0.0, 0.0, 0.0]),   # stiff XY, compliant Z
        "kp_rot": np.array([20.0, 20.0, 10.0]),
        "description": "Initial contact: stiff lateral, compliant normal",
    },
    "wipe_x": {
        "log_kp": np.array([5.0, 3.0, 2.5, 0.0, 0.0, 0.0]),   # stiff along stroke
        "kp_rot": np.array([25.0, 10.0, 10.0]),
        "description": "Wiping in X: high X stiffness, compliant normal",
    },
    "wipe_y": {
        "log_kp": np.array([3.0, 5.0, 2.5, 0.0, 0.0, 0.0]),
        "kp_rot": np.array([10.0, 25.0, 10.0]),
        "description": "Wiping in Y: high Y stiffness, compliant normal",
    },
    "lift": {
        "log_kp": np.array([3.0, 3.0, 4.0, 0.0, 0.0, 0.0]),   # stiff Z for lift
        "kp_rot": np.array([15.0, 15.0, 15.0]),
        "description": "Lifting off surface: stiffen Z",
    },
}

SYSTEM_PROMPT = """You are an impedance control expert for a robot wiping task.
The robot uses a Variable Impedance Controller on an SE(3) Riemannian manifold.
Your job: given the current task state, select the optimal impedance mode.

Available modes: approach, contact_edge, wipe_x, wipe_y, lift

Respond with ONLY the mode name, nothing else."""

@dataclass  
class ImpedanceSuggestion:
    mode: str
    log_kp_prior: np.ndarray   # 6-dim Mandel log-space suggestion
    kp_rot_prior: np.ndarray   # 3-dim rotational Kp suggestion
    confidence: float           # 0-1, how much to weight prior vs RL residual


BACKEND_DEFAULTS = {
    "openai": "gpt-4o-mini",   # cheap + fast; swap to "gpt-4o" for better reasoning
    "ollama": "llama3.2",      # popular in robot learning; `ollama pull llama3.2`
}


class LLMImpedancePlanner:
    """
    Periodically queries an LLM to select an impedance mode based on
    task state description. Between queries, holds last suggestion.
    """
    def __init__(
        self,
        backend: str = "openai",        # "openai" | "ollama"
        model: str | None = None,       # defaults per BACKEND_DEFAULTS
        query_every_n_steps: int = 50,
        prior_weight: float = 0.4,      # blend: action = prior_weight*hint + (1-w)*rl
    ):
        self.model = model or BACKEND_DEFAULTS[backend]
        self.query_every = query_every_n_steps
        self.prior_weight = prior_weight
        self._step = 0
        self._last: ImpedanceSuggestion = self._default_suggestion()
        self._history: list[dict] = []

        if backend == "ollama":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:  # openai — reads OPENAI_API_KEY from env
            self.client = OpenAI()

    def _default_suggestion(self) -> ImpedanceSuggestion:
        m = IMPEDANCE_MODES["approach"]
        return ImpedanceSuggestion("approach", m["log_kp"], m["kp_rot"], 0.3)

    def _build_state_description(self, obs_dict: dict) -> str:
        """Convert raw obs dict to a text description for the LLM."""
        eef_pos = obs_dict.get("robot0_eef_pos", np.zeros(3))
        force = obs_dict.get("robot0_eef_force", np.zeros(3))  # if available
        wipe_pct = obs_dict.get("wipe_pct", 0.0)
        
        lines = [
            f"EEF position (xyz): [{eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}]",
            f"Contact force (xyz): [{force[0]:.2f}, {force[1]:.2f}, {force[2]:.2f}] N",
            f"Wipe completion: {wipe_pct*100:.1f}%",
        ]
        
        # Add table height context if available
        table_z = obs_dict.get("table_z", 0.8)
        clearance = eef_pos[2] - table_z
        lines.append(f"EEF clearance above table: {clearance:.4f} m")
        
        return "\n".join(lines)

    def query(self, obs_dict: dict) -> ImpedanceSuggestion:
        """Query LLM for impedance mode. Respects query_every throttle."""
        self._step += 1
        if self._step % self.query_every != 0:
            return self._last

        state_desc = self._build_state_description(obs_dict)
        user_msg = f"Current robot state:\n{state_desc}\n\nWhich impedance mode?"

        # Use multi-turn for context continuity within episode
        self._history.append({"role": "user", "content": user_msg})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=20,
            temperature=0,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._history,
        )
        mode_str = response.choices[0].message.content.strip().lower()
        
        # Validate and fallback
        if mode_str not in IMPEDANCE_MODES:
            mode_str = "approach"
        
        self._history.append({"role": "assistant", "content": mode_str})
        # Keep history bounded (last 10 turns)
        if len(self._history) > 20:
            self._history = self._history[-20:]

        m = IMPEDANCE_MODES[mode_str]
        self._last = ImpedanceSuggestion(
            mode=mode_str,
            log_kp_prior=m["log_kp"].copy(),
            kp_rot_prior=m["kp_rot"].copy(),
            confidence=self.prior_weight,
        )
        return self._last

    def reset(self):
        """Call at episode start to reset conversation context."""
        self._history = []
        self._last = self._default_suggestion()
        self._step = 0