from .tilted_wipe import TiltedWipe
from .delta_wipe import DeltaWipe
from robosuite.environments.base import register_env

register_env(TiltedWipe)
register_env(DeltaWipe)
