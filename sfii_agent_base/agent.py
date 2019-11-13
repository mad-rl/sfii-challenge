from .knowledge import Knowledge
from .interpreter import Interpreter
from .actuator import Actuator
from .experiences import Experiences


"""
Default agent parameters:

    'frames': 16,
    'cnn_channels': 32,
    'n_outputs': 49,
    'screen_height': 256,
    'screen_width': 200,
    'width': 80,
    'height': 80,
    'start_from_model': "models/sf2_a3c.pth",
    'module': "src.environments.gym_retro.my_agent.agent",
    "class": "Agent"
    
"""


class Agent():
    def __init__(self, parameters):
        self.action_space = parameters['n_outputs']
        self.input_frames = parameters['frames']
        self.width = parameters['width']
        self.height = parameters['height']

        self.knowledge = Knowledge(self.input_frames, self.action_space)
        self.interpreter = Interpreter(
            frames=self.input_frames, width=self.width, height=self.height)
        self.actuator = Actuator()
        self.experiences = Experiences()

        self.total_steps = 0
        self.max_reward = 0
        self.rewards = []

    def load_model(self, model):
        self.knowledge.model.load_state_dict(model.state_dict())

    def get_model(self):
        return self.knowledge.get_model()

    def initialize_optimizer(self, shared_agent):
        self.knowledge.initialize_optimizer(shared_agent)

    def get_state(self, state, observation):
        new_state = self.interpreter.obs_to_state(state, observation)

        return new_state

    def get_action(self, state):
        action, value = self.knowledge.get_action(state)

        return self.actuator.agent_to_env(action), value

    def calculate_reward(self, player_health, enemy_health, life_value=176):
        health_gap = player_health - enemy_health
        reward = float(health_gap / life_value)

        return reward

    def add_experience(self, state, env_action, reward, next_state, info=None):
        reward_by_health = self.calculate_reward(
            info['health'], info['enemy_health'])
        agent_action = self.actuator.env_to_agent(env_action)

        self.experiences.add(state, agent_action, reward_by_health, next_state)
        self.rewards.append(reward)

    def start_step(self, current_step):
        pass

    def end_step(self, current_step):
        self.episode_steps = self.episode_steps + 1
        self.total_steps = self.total_steps + 1

    def start_episode(self, current_episode):
        self.episode_steps = 0

    def end_episode(self, current_episode):
        self.rewards = []

    def train(self, game_finished, shared_agent):
        self.knowledge.train(
            self.experiences.get(), game_finished, shared_agent)
        self.experiences.reset()
