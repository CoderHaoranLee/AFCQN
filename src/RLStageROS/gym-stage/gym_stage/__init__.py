import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='StageICRALidarNnEnv-v0',
    entry_point='gym_stage.envs.robots:StageICRALidarNNEnv',
    max_episode_steps=1000,
    # More arguments here
)

register(
    id='StageMazeCarLidarNnEnv-v0',
    entry_point='gym_stage.envs.robots:StageMazeCarLidarNNEnv',
    max_episode_steps=1000,
    # More arguments here
)

register(
    id='StageCampCarLidarNnEnv-v0',
    entry_point='gym_stage.envs.robots:StageCampCarLidarNNEnv',
    max_episode_steps=1000,
    # More arguments here
)