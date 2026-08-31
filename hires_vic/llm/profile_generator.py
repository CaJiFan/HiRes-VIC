"""
LLM-based impedance profile generator for novel manipulation tasks.

Given a task specification, queries an LLM (GPT-4o) to generate:
  - Named impedance control phases with physical stiffness values
  - A tailored system prompt for the runtime LLMImpedancePlanner

The output YAML is saved to the specified path and can be used directly by
LLMImpedancePlanner via the `profile_path` argument.

Usage (standalone script)
─────────────────────────
  python -m hires_vic.llm.profile_generator \\
      --task PegInsertionSide-v1 \\
      --out configs/pih_impedance_profile.yaml

  # Custom task description:
  python -m hires_vic.llm.profile_generator \\
      --task MyCustomEnv \\
      --description "Robot must open a door by pulling a handle" \\
      --insertion-axis X \\
      --out configs/door_impedance_profile.yaml

Note: requires OPENAI_API_KEY in environment or .env file.
"""

from __future__ import annotations

import argparse
import os
import sys
import yaml
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()


# ── Built-in task specifications ─────────────────────────────────────────────

TASK_SPECS: dict[str, dict] = {
    "PegInsertionSide-v1": {
        "name": "PegInsertionSide-v1",
        "robot": "7-DOF Panda arm with parallel-jaw gripper",
        "description": (
            "Peg-in-hole (PiH) insertion from the side. "
            "The peg is grasped by (or rigidly attached to) the robot EEF. "
            "The cylindrical hole is fixed in the world. "
            "The robot must align the peg tip with the hole opening, then "
            "push it to full insertion depth."
        ),
        "insertion_axis": "Z (roughly, in EEF task space)",
        "key_challenges": [
            "sub-millimetre lateral alignment of peg tip with hole",
            "compliant insertion to avoid jamming when misaligned",
            "sufficient axial stiffness for depth progress",
            "final seating to full depth without excessive force",
        ],
        "available_obs": [
            "EEF position (xyz)",
            "peg head position relative to hole opening",
            "binary contact flag",
            "estimated insertion depth (0 = not inserted, 1 = fully seated)",
        ],
        "success_criteria": "peg fully seated inside the hole (depth ≥ threshold)",
        "suggested_phases": ["approach", "align", "insert", "seated"],
    },
    "StackCube-v1": {
        "name": "StackCube-v1",
        "robot": "7-DOF Panda arm with parallel-jaw gripper",
        "description": (
            "Pick up a red cube and stack it on top of a blue cube. "
            "Requires grasping, lifting, and precise placement."
        ),
        "insertion_axis": "Z (vertical placement)",
        "key_challenges": [
            "stable grasp acquisition",
            "compliant contact during placement to avoid toppling",
            "lateral precision for stacking",
        ],
        "available_obs": [
            "EEF position (xyz)",
            "red cube position",
            "blue cube position",
            "gripper state",
        ],
        "success_criteria": "red cube stably resting on top of blue cube",
        "suggested_phases": ["approach", "grasp", "lift", "place"],
    },
    # ── TiltedWipe (45°) ──────────────────────────────────────────────────────
    # Factual inputs only — no pre-solved thresholds or axis-specific answers.
    # Let the LLM reason about geometry, compliance, and phase structure itself.
    "TiltedWipe": {
        "name": "TiltedWipe (Robosuite, 45° tilt)",
        "robot": "7-DOF Panda arm with a flat wiping tool rigidly attached to the EEF",
        "description": (
            "The Panda arm must use the wiping tool to clear N dirt markers "
            "from a flat table surface that is tilted 45° around the world Y-axis. "
            "The robot starts above the table and must descend to make contact, "
            "then sweep the tool across all marker locations to achieve full coverage."
        ),
        "geometry": (
            "Table tilt: 45° pitch around the world Y-axis. "
            "The impedance controller is expressed in the world frame: "
            "stiffness is a diagonal [Kx, Ky, Kz] in world-X, world-Y, world-Z axes."
        ),
        "key_challenges": [
            "Maintaining sustained surface contact on the tilted table throughout the episode",
            "Generating enough lateral force authority to drive the wiping stroke across markers",
            "Avoiding excessive contact force that could cause tool bounce or joint limit violations",
            "The table is not horizontal — the contact normal is not aligned with world-Z",
        ],
        "reward_signals": (
            "The reward combines: a per-step bonus for maintaining surface contact (binary — "
            "zero if not touching the table), a bonus for each new marker wiped (only while "
            "in contact), a reaching bonus proportional to proximity to unwiped markers, "
            "a large completion bonus if all markers are cleared, and penalties for "
            "excessive contact force and jerky end-effector motion."
        ),
        "motion_pattern": (
            "The workspace is small enough that all markers are reachable in a single "
            "continuous sweeping stroke. The expected trajectory is: descend to the surface "
            "(no contact), establish contact, then sweep continuously without lifting off. "
            "A mid-episode 'lift' or 'reposition' phase does not occur in practice and "
            "would be observationally indistinguishable from the initial approach."
        ),
        "available_obs": [
            "EEF position in world frame (xyz)",
            "Surface contact flag (True/False)",
            "Wipe completion fraction [0.0, 1.0]",
            "Distance to centroid of remaining unwiped markers (metres)",
        ],
        "success_criteria": "All N dirt markers cleared within the episode horizon.",
        "suggested_phases": ["approach", "transition", "wipe"],
    },
    "TiltedWipeHQ": {
        "name": "TiltedWipe (Robosuite, 45° tilt)",
        "robot": "7-DOF Panda arm with a flat wiping tool rigidly attached to the EEF",
        "description": (
            "The Panda arm uses a flat wiping tool to clear sequential dirt markers from a "
            "whiteboard tilted 45° around the world Y-axis. The EEF is initialized ~15 cm "
            "above the board with a +45° pitch rotation so its flat bottom is parallel to the board."
        ),
        "geometry": (
            "Table tilt: 45° pitch around the world Y-axis. "
            "Surface normal in world frame: [0.707, 0, 0.707]. "
            "The impedance controller is expressed in the world frame: "
            "stiffness is [Kx, Ky, Kz] in world-X, world-Y, world-Z axes."
        ),
        "physical_dynamics": (
            "Contact Force Scale: Clearing dirt markers and satisfying the force tracking reward "
            "requires maintaining ~15 N of contact force against the surface. In impedance control, "
            "this corresponds to ~1–2 cm of commanded virtual penetration with normal stiffness in "
            "the 100–150 N/m range. Excessively soft normal stiffness (<30 N/m) fails to clear markers.\n"
            "Tool Orientation: The flat pad must remain parallel to the 45° incline during sweeping; "
            "rotational stiffness should be firm (120–180 N·m/rad) to prevent torque-induced pad twisting.\n"
            "Tangential Motion: Overcoming surface friction to drive the wiping stroke across markers "
            "requires high tangential stiffness (120–180 N/m)."
        ),
        "available_obs": [
            "EEF position in world frame (xyz)",
            "Surface contact flag (True/False)",
            "Relative 3D vector to active target marker [dx, dy, dz]",
            "Distance to active target marker (metres)",
            "Individual marker wiped flags (True/False)",
            "Overall wipe completion fraction [0.0, 1.0]",
        ],
        "success_criteria": "All dirt markers cleared on the 45° tilted whiteboard within horizon.",
        "suggested_phases": ["approach", "wipe"],
    },

    # ── TiltedWipev3 (60°) ───────────────────────────────────────────────────
    # Steeper tilt: surface normal is more asymmetric [0.866, 0, 0.5].
    # Contact normal is predominantly in world-X, making axis-specific compliance
    # significantly harder to get right — designed to stress-test Riemannian geometry.
    "TiltedWipev3": {
        "name": "TiltedWipev3 (Robosuite, 60° tilt)",
        "robot": "7-DOF Panda arm with a flat wiping tool rigidly attached to the EEF",
        "description": (
            "The Panda arm must use the wiping tool to clear N dirt markers "
            "from a flat table surface that is steeply tilted 60° around the world Y-axis. "
            "The robot starts above the table and must descend to make contact, "
            "then sweep the tool across all marker locations to achieve full coverage. "
            "The steep tilt makes contact normal force management more demanding than "
            "at 45° and requires careful axis-specific impedance tuning."
        ),
        "geometry": (
            "Table tilt: 60° pitch around the world Y-axis. "
            "The impedance controller is expressed in the world frame: "
            "stiffness is a diagonal [Kx, Ky, Kz] in world-X, world-Y, world-Z axes."
        ),
        "key_challenges": [
            "Maintaining sustained surface contact on a steeply tilted (60°) surface",
            "The steep tilt means contact-normal compliance is more strongly required "
            "in world-X than in world-Z — asymmetric axis requirements",
            "Generating enough lateral stroke force while conforming to the tilted surface",
            "Avoiding excessive contact force or arm instability due to the steep angle",
            "The contact normal is far from world-Z at 60° tilt",
        ],
        "reward_signals": (
            "Same reward structure as TiltedWipe (45°): per-step surface contact bonus "
            "(binary), per-marker wipe bonus (only while in contact), proximity reaching "
            "bonus, completion bonus, and penalties for excessive force and jerky motion."
        ),
        "motion_pattern": (
            "Same as TiltedWipe (45°): single continuous sweep without lifting off. "
            "Expected trajectory: descend → establish contact → sweep continuously. "
            "No mid-episode repositioning phase occurs."
        ),
        "available_obs": [
            "EEF position in world frame (xyz)",
            "Surface contact flag (True/False)",
            "Wipe completion fraction [0.0, 1.0]",
            "Distance to centroid of remaining unwiped markers (metres)",
        ],
        "success_criteria": "All N dirt markers cleared within the episode horizon.",
        "suggested_phases": ["approach", "transition", "wipe"],
    },
    "Door": {
        "name": "Door (Robosuite)",
        "robot": "7-DOF Panda arm with parallel-jaw gripper",
        "description": (
            "The robot must open a door by grasping a handle and pulling/rotating the door "
            "open. The door swings on a vertical hinge. Success requires rotating the handle "
            "latch (joint angle > 0.25 rad) and then pulling the door open past the target angle."
        ),
        "insertion_axis": "X (primary outward pulling axis)",
        "geometry": "Door swings around vertical Z hinge axis. Handle latch rotates in YZ plane.",
        "key_challenges": [
            "Precision reaching and alignment with the door handle",
            "Sufficient grasp and rotational compliance to turn the spring-loaded latch",
            "High pulling stiffness along normal with compliance along circular sweep arc",
        ],
        "available_obs": [
            "EEF position in world frame (xyz)",
            "Surface contact flag (True/False)",
            "Door frame position (xyz)",
            "Handle position (xyz)",
            "Vector from EEF to handle (xyz) and Euclidean distance",
            "Handle rotated angle (qpos, rad)",
            "Door open hinge angle (qpos, rad)",
        ],
        "success_criteria": "Door opened past target angle (~0.35 rad).",
        "suggested_phases": ["approach", "grasp_turn", "pull_open", "swing_through"],
    },
}

GENERATOR_SYSTEM_PROMPT = """\
You are an expert in robot impedance control and Variable Impedance Control (VIC).

You will be given a description of a robotic manipulation task and must output
a YAML impedance profile with exactly two top-level keys:

1. `phases` — a dict of named phases, each with:
   - kp_trans: [Kx, Ky, Kz]  # translational stiffness, N/m, range 1–300
   - kp_rot:   [Kx, Ky, Kz]  # rotational stiffness, N·m/rad, range 1–300
   - description: one-line explanation of the impedance rationale

2. `planner_prompt` — a multi-line string used as the system prompt for a
   runtime LLM that selects among the phases step-by-step during RL training.
   It must list the available mode names and describe when to use each.

Rules:
- Provide 3–5 phases that cover the task's contact-rich sub-tasks.
- Choose stiffness values based on physical reasoning (e.g., compliance
  perpendicular to a contact surface, stiffness along the action axis).
- kp values must be in [1, 300]. Use lower values for compliant axes
  and higher for precision/force axes.
- The planner_prompt must be self-contained and tell the runtime LLM to
  respond with ONLY the mode name, nothing else.
- Output ONLY valid YAML — no prose, no markdown fences, no extra keys.
"""


def _build_user_message(spec: dict) -> str:
    lines = [
        f"Task: {spec['name']}",
        f"Robot: {spec['robot']}",
        f"Description: {spec['description']}",
        f"Primary action axis: {spec.get('insertion_axis', 'Z')}",
    ]

    if "geometry" in spec:
        lines.append(f"Geometry / coordinate frame: {spec['geometry']}")

    lines.append("Key challenges:")
    for c in spec.get("key_challenges", []):
        lines.append(f"  - {c}")

    if "physical_failure_modes" in spec:
        lines.append("Physical failure modes to avoid:")
        for f in spec["physical_failure_modes"]:
            lines.append(f"  ⚠ {f}")

    if "reward_signals" in spec:
        lines.append(f"Reward signals: {spec['reward_signals']}")

    if "hardware_limits" in spec:
        lines.append(f"Hardware limits: {spec['hardware_limits']}")

    lines.append("Available observations at runtime:")
    for o in spec.get("available_obs", []):
        lines.append(f"  - {o}")

    lines.append(f"Success criteria: {spec.get('success_criteria', 'task complete')}")

    if "motion_pattern" in spec:
        lines.append(f"Motion pattern / trajectory constraints: {spec['motion_pattern']}")

    lines.append(f"Suggested phase names: {spec.get('suggested_phases', [])}")

    if "phase_hints" in spec:
        lines.append("Per-phase impedance hints:")
        for phase, hint in spec["phase_hints"].items():
            lines.append(f"  [{phase}]: {hint}")

    lines.append("\nGenerate the impedance profile YAML now.")
    return "\n".join(lines)



def generate_profile(
    task_name: str,
    custom_spec: dict | None = None,
    model: str = "gpt-4o",
    out_path: str | None = None,
) -> dict:
    """
    Query an LLM to generate an impedance profile for `task_name`.

    Parameters
    ----------
    task_name   : Key in TASK_SPECS or any string (used with custom_spec).
    custom_spec : Override or supplement the built-in task spec dict.
    model       : OpenAI model to use (default: gpt-4o for best reasoning).
    out_path    : If provided, write the resulting YAML to this path.

    Returns
    -------
    dict : Parsed profile (same structure as configs/pih_impedance_profile.yaml).
    """
    spec = dict(TASK_SPECS.get(task_name, {"name": task_name}))
    if custom_spec:
        spec.update(custom_spec)

    client = OpenAI()
    print(f"[profile_generator] Querying {model} for task: {task_name} ...")

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        max_tokens=1500,
        messages=[
            {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(spec)},
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model ignored instructions
    if raw.startswith("```"):
        raw = "\n".join(
            line for line in raw.splitlines()
            if not line.startswith("```")
        )

    profile = yaml.safe_load(raw)

    # Inject metadata so the file is self-documenting
    full_profile = {
        "task": task_name,
        "robot": spec.get("robot", "Panda"),
        "description": spec.get("description", ""),
        "phases": profile.get("phases", {}),
        "planner_prompt": profile.get("planner_prompt", ""),
    }

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            yaml.dump(full_profile, f, default_flow_style=False, allow_unicode=True,
                      default_style=None, sort_keys=False)
        print(f"[profile_generator] Saved to {out_path}")

    return full_profile


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Generate LLM impedance profile for a task.")
    p.add_argument("--task", default="PegInsertionSide-v1",
                   help="Task name (built-in key or custom string)")
    p.add_argument("--description", default=None,
                   help="Override task description (for custom tasks)")
    p.add_argument("--insertion-axis", default=None, dest="insertion_axis",
                   help="Primary action axis (e.g. X, Y, Z)")
    p.add_argument("--out", default=None,
                   help="Output YAML path (default: configs/<task>_impedance_profile.yaml)")
    p.add_argument("--model", default="gpt-4o",
                   help="OpenAI model to use (default: gpt-4o)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    out = args.out or f"configs/{args.task.lower().replace('-', '_')}_impedance_profile.yaml"

    custom = {}
    if args.description:
        custom["description"] = args.description
    if args.insertion_axis:
        custom["insertion_axis"] = args.insertion_axis

    profile = generate_profile(
        task_name=args.task,
        custom_spec=custom if custom else None,
        model=args.model,
        out_path=out,
    )

    print("\n── Generated profile ──────────────────────────────────────")
    print(yaml.dump(profile, default_flow_style=False, allow_unicode=True, sort_keys=False))
