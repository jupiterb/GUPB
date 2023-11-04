from abc import ABC, abstractmethod
from dataclasses import dataclass
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
import stable_baselines3.dqn as dqn
from threading import Thread

from gupb.controller.batman.rl.environment import GUPBEnv
from gupb.controller.batman.rl.environment.feature_extractors import NeighborhoodCNN


@dataclass
class AlgoConfig:
    learning_rate: float = 0.005
    batch_size: int = 32
    buffer_size: int = 10000
    learning_starts: int = 300
    tau: float = 0.05
    gamma: float = 0.98
    feature_extractor_class: BaseFeaturesExtractor = NeighborhoodCNN


class SomeAlgo(ABC):
    def __init__(self, env: GUPBEnv, config: AlgoConfig) -> None:
        self._env = env
        self._algo = self._build_algo(env, config)
        self._training = None
        self._terminated = False

    def run(self, from_timestep: int = 0, to_timesteps: int = 100) -> None:
        self._terminated = False
        self._total_timesteps = to_timesteps
        self._training = Thread(target=self._run_training)
        self._training.start()
        self._algo.num_timesteps = from_timestep

    def terminate(self) -> int:
        """Rturns algo timesteps"""
        if self._training is None:
            return 0
        self._terminated = True
        timesetps = self._algo.num_timesteps
        self._algo.num_timesteps = self._total_timesteps
        while self._training.is_alive():
            self._env.stop_waiting()
        self._training.join()
        return timesetps

    def save(self, path: str) -> None:
        self._algo.save(path)
        self._algo.save_replay_buffer(f"{path}_reply_buffer")

    def load(self, path: str) -> None:
        self._algo.load(path)
        self._algo.load_replay_buffer(f"{path}_reply_buffer")

    @abstractmethod
    def _build_algo(self, env, config: AlgoConfig) -> OffPolicyAlgorithm:
        raise NotImplementedError()

    def _run_training(self) -> None:
        try:
            self._algo.learn(self._total_timesteps)
        except:
            pass


class DQN(SomeAlgo):
    def _build_algo(self, env, config: AlgoConfig) -> OffPolicyAlgorithm:
        return dqn.DQN(
            policy=dqn.CnnPolicy,
            env=env,
            learning_rate=config.learning_rate,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            learning_starts=config.learning_starts,
            tau=config.tau,
            gamma=config.gamma,
            verbose=1,
            policy_kwargs={
                "features_extractor_class": config.feature_extractor_class,
            },
        )
