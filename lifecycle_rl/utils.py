'''
Utils
'''

import gym
from gym import spaces, logger, utils, error
from gym.utils import seeding
#from stable_baselines.common import set_global_seeds

def make_env(env_id, rank, kwargs, seed=None, use_monitor=True):
    """
    Utility function for multiprocessed env.#

    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id,kwargs=kwargs)
        if seed is not None:
            env.seed(seed + rank)
            env.env_seed(seed + rank + 100)

        # monitor enables various things, not used by default
        #print('monitor=',use_monitor)
        #if use_monitor:
        #    env = Monitor(env, self.log_dir, allow_early_resets=True)

        return env

#    if seed is not None:
#        set_global_seeds(seed)

    return _init()
