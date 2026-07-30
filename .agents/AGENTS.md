
## SPD Manifold Action Space Processing

**CRITICAL RULE:** Whenever editing `geometric.py` (especially the `step()` function), you MUST ensure that SPD runs (`use_spd_manifold=True`) always learn and process the FULL 6D Mandel basis for translational stiffness, regardless of whether `use_llm_prior` is enabled or disabled. 

Specifically:
- The action space must allocate 9 dimensions for stiffness (6 for translational SPD Mandel basis, 3 for rotational stiffness).
- Both the LLM SPD path and the pure RL (no-LLM) SPD path must construct a 6D tensor for the GRL mapping (`spd_grl_map`), where `action[0:3]` maps to the diagonal terms and `action[3:6]` maps to the off-diagonal (coupling) terms.
- NEVER reduce the pure RL SPD path to a 3D diagonal-only learning space.

