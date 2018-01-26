import gym
from gym import spaces
import random


class BitFlip(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 30
    }

    def __init__(self, bit_length=16, max_steps=None):
        super(BitFlip, self).__init__()
        self.bit_length = bit_length

        if max_steps is None:
            # default to 2x bit_length
            self.max_steps = bit_length * 2
        elif max_steps == 0:
            self.max_steps = None
        else:
            self.max_steps = max_steps

        # spaces documentation: https://gym.openai.com/docs/
        self.action_space = spaces.Discrete(bit_length)
        self.observation_space = spaces.Dict({
            'state': spaces.Box(low=0, high=1, shape=(bit_length, )),
            'goal': spaces.Box(low=0, high=1, shape=(bit_length, )),
        })

        self.reset()

    def _terminate(self):
        return self.state == self.target or self.steps >= self.max_steps

    def _reward(self):
        return -1 if self.state != self.target else 0

    def _step(self, action):
        # action an int (0, self.bit_length)
        self.state[action] = not self.state[action]
        self.steps += 1

        return (self._get_obs(), self._reward(), self._terminate(), {})

    def _reset(self):
        self.steps = 0
        self.state = [random.choice([1, 0]) for _ in self.bit_length]
        self.target = [random.choice([1, 0]) for _ in self.bit_length]

        return self._get_obs()

    def _get_obs(self):
        return self.state

    def _render(self, mode='human', close=False):
        pass
