import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from multi_evo_sim.agents.base_agent import BaseAgent, ActionType
from multi_evo_sim.agents.neural_agent import NeuralAgent


def test_base_agent_act():
    agent = BaseAgent()
    action = agent.act({'resource_here': 1})
    assert action.type == ActionType.GATHER

    action = agent.act({'resource_here': 0})
    assert action.type == ActionType.MOVE


def test_neural_agent_act_returns_action():
    agent = NeuralAgent(input_size=2)
    obs = {
        'position': (1, 1),
        'resource_here': 0,
        'inventory': 0,
        'resources': [],
        'danger': False,
    }
    action = agent.act(obs)
    assert action.type in {ActionType.MOVE, ActionType.GATHER, ActionType.COOPERATE}
