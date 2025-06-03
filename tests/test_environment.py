import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multi_evo_sim.env.world import World
from multi_evo_sim.env.resource import Resource
from multi_evo_sim.agents.base_agent import BaseAgent


def test_world_step_gather():
    world = World(width=3, height=3, resources=[Resource((0, 0))], resource_regen=False)
    agent = BaseAgent()
    world.add_agent(agent, position=(0, 0))
    world.step()
    assert agent.inventory == 1
    assert world.resources[0].consumed is True
    assert world.agents[0][1] == (0, 0)
