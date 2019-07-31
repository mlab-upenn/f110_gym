from f110_gym.f110_core import f110Env, f110ActionWrapper, f110ObservationWrapper, f110Wrapper

__author__ = "Dhruv Karthik <dhruvkar@seas.upenn.edu>"

class SkipEnv(f110Wrapper):
    def __init__(self, env, skip=4):
        """Return only 'skip-th frame"""
        f110Wrapper.__init__(self, env)
        self._skip = skip
        
    def step(self, action):
        """Repeat action & sum reward"""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def serialize_obs(self):
        return self.env.serialize_obs()

class PreprocessImg(f110ObservationWrapper):
    import cv2

    def __init__(self, env):
        f110ObservationWrapper.__init__(self, env)
        self.observation_space = self.env.observation_space

    def observation(self, obs):
        """ For now, Crop any 'img' observations, in future, add input funclist array to preprocess"""
        new_obs = obs
	src_img = obs["img"]
	new_obs["img"] = src_img[100:200, :, :]
        return new_obs

    def serialize_obs(self):
        return self.env.serialize_obs()

def make_imitation_env(skip=10):
    env = f110Env()
    env = PreprocessImg(env)
    env = SkipEnv(env, skip=skip)
    return env
