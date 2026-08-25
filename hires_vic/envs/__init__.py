from .tilted_wipe import TiltedWipe
from .delta_wipe import DeltaWipe
# from .scattered_wipe_arena import ScatteredWipeArena
from robosuite.environments.base import register_env

print("Registering custom environments...")
register_env(TiltedWipe)
# register_env(ScatteredWipeArena)
register_env(DeltaWipe)
