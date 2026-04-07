import gymnasium as gym


class WipeMetricWrapper(gym.Wrapper):
    def step(self, action):
        step_returns = self.env.step(action)

        if len(step_returns) == 4:
            # print('we using legacy Gym')
            obs, reward, done, info = step_returns
            terminated = done
            truncated = False # Legacy gym didn't separate these natively
        elif len(step_returns) == 5:
            # print('we using Gymnasium')
            # print(self.env.env.action_spec, self.env.env.observation_spec)
            obs, reward, terminated, truncated, info = step_returns
        else:
            raise ValueError(f"Unexpected number of returns from environment step: {len(step_returns)}")
       

        raw_env = self.env.env

        total_markers = raw_env.num_markers
        wiped_markers = len(raw_env.wiped_markers)
        percent_wiped = wiped_markers / total_markers

        info["physics/raw_wipe_percentage"] = percent_wiped
            
        return obs, reward, terminated, truncated, info