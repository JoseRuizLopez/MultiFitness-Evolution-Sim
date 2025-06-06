import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multi_evo_sim.env.world import World
from multi_evo_sim.env.resource import Resource
from multi_evo_sim.agents.base_agent import BaseAgent
from multi_evo_sim.agents.base_agent import Action, ActionType


def test_world_step_gather():
    world = World(width=3, height=3, resources=[Resource((0, 0))], resource_regen=False)
    agent = BaseAgent()
    world.add_agent(agent, position=(0, 0))
    world.step()
    assert agent.inventory == 1
    assert world.resources[0].consumed is True
    assert world.agents[0][1] == (0, 0)


def test_last_coop_updates_on_cooperate():
    """Comprueba que tras cooperar se actualiza ``world.last_coop``."""

    class CoopAgent(BaseAgent):
        def act(self, _obs):
            return Action(ActionType.COOPERATE)

    world = World(width=3, height=3, resource_regen=False)
    agent1 = CoopAgent()
    agent1.inventory = 1
    agent2 = BaseAgent()
    world.add_agent(agent1, position=(1, 1))
    world.add_agent(agent2, position=(1, 1))
    world.step()
    assert world.last_coop == (1, 1)
