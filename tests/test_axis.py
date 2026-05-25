from robosuite.utils import transform_utils as T
import numpy as np


raw = env.unwrapped._get_observations()
q = np.asarray(raw['robot0_eef_quat'])
q = q / (np.linalg.norm(q) + 1e-12)
R = T.quat2mat(q)

# local +Z expressed in world coords:
print('EEF +Z (world):', R[:, 2])
