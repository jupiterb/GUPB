from typing import Optional

from gupb import controller
from gupb.model import arenas
from gupb.controller.batman.rl.algo import DQN, AlgoConfig
from gupb.controller.batman.rl.environment import GUPBEnv
from gupb.controller.batman.rl.environment.observation import SimpleObservation
from gupb.controller.batman.knowledge.knowledge import Knowledge
from gupb.controller.batman.utils.observer import Observer, Observable
from gupb.model.characters import Action, ChampionKnowledge, Tabard


class BatmanController(controller.Controller, Observer[Action], Observable[Knowledge]):
    def __init__(self, name: str) -> None:
        super().__init__()
        Observer.__init__(self)
        Observable.__init__(self)

        self._name = name
        self._episode = 0
        self._game = 0
        self._knowledge: Optional[Knowledge] = None

        observation = SimpleObservation(20)
        self._env = GUPBEnv(observation)

        self._env.attach(self)
        self.attach(self._env)

        self._algo = DQN(self._env, AlgoConfig())

        self._timestep = 0
        self._timesteps_per_game = 10000

    def decide(self, knowledge: ChampionKnowledge) -> Action:
        assert (
            self._knowledge is not None
        ), "Reset was not called before first decide() call"

        self._episode += 1
        self._knowledge.update(knowledge, self._episode)
        self.observable_state = self._knowledge
        action = self.wait_for_observed()
        return action

    def praise(self, score: int) -> None:
        self._timestep = self._algo.terminate()
        self._algo.save()

    def reset(self, game_no: int, arena_description: arenas.ArenaDescription) -> None:
        try:
            self._algo.load()
        except:
            pass
        self._algo.run(self._timestep, self._timestep + self._timesteps_per_game)

        self._episode = 0
        self._game += game_no
        self._knowledge = Knowledge(arena_description)
        self.observable_state = self._knowledge

    @property
    def name(self) -> str:
        return self._name

    @property
    def preferred_tabard(self) -> Tabard:
        return Tabard.STRIPPED  # TODO change to batman
