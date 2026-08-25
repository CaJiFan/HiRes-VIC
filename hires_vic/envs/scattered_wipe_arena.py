# hires_vic/envs/scattered_wipe_arena.py
import numpy as np
from robosuite.models.arenas import WipeArena

class ScatteredWipeArena(WipeArena):
    """
    WipeArena variant where markers are placed uniformly at random across the
    table surface, instead of along a random-walk path. This ensures markers
    are spread out even with small num_markers (e.g., 8-10 waypoints).
    """
    def sample_path_pos(self, pos):
        # Ignore prev pos entirely — sample independently
        return np.array((
            self.rng.uniform(
                -self.table_half_size[0] * self.coverage_factor + self.line_width / 2,
                 self.table_half_size[0] * self.coverage_factor - self.line_width / 2,
            ),
            self.rng.uniform(
                -self.table_half_size[1] * self.coverage_factor + self.line_width / 2,
                 self.table_half_size[1] * self.coverage_factor - self.line_width / 2,
            ),
        ))
