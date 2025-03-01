from __future__ import annotations

import argparse
import uuid

import hockey.hockey_env as h_env
import numpy as np

from comprl.client import Agent, launch_client

import sys
import os

# append path to Hockey_training
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
#sys.path.append("C:/Users/pi030/Documents/Uni/ReinforcementLearning/finalProject/code/RL_Project/DDPG_Hockey")

import Hockey_training.training as training


class RandomAgent(Agent):
    """A hockey agent that simply uses random actions."""

    def get_step(self, observation: list[float]) -> list[float]:
        return np.random.uniform(-1, 1, 4).tolist()

    def on_start_game(self, game_id) -> None:
        print("game started")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


class HockeyAgent(Agent):
    """A hockey agent that trained with different algorithms."""

    def __init__(self, type: str) -> None:
        super().__init__()
        #self.hockey_agent = h_env.BasicOpponent(weak=weak)
        if type == "TD3":
            self.hockey_agent = training.TD3Opponent()
        elif type == "RND":
            self.hockey_agent = training.RNDOpponent()
        elif type == "pinkNoise":
            self.hockey_agent = training.PinkNoiseOpponent()
        elif type == "combi":
            self.hockey_agent = training.CombiOpponent()
        else:
            raise ValueError(f"Unknown agent: {type}")
        
        print(f"agent initialized...")

    def get_step(self, observation: list[float]) -> list[float]:

        action = self.hockey_agent.act(observation).tolist()

        return action

    def on_start_game(self, game_id) -> None:
        game_id = uuid.UUID(int=int.from_bytes(game_id, byteorder='big'))
        print(f"Game started (id: {game_id})")

    def on_end_game(self, result: bool, stats: list[float]) -> None:
        text_result = "won" if result else "lost"
        print(
            f"Game ended: {text_result} with my score: "
            f"{stats[0]} against the opponent with score: {stats[1]}"
        )


# Function to initialize the agent.  This function is used with `launch_client` below,
# to lauch the client and connect to the server.
def initialize_agent(agent_args: list[str]) -> Agent:
    # Use argparse to parse the arguments given in `agent_args`.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=["TD3", "RND", "pinkNoise", "combi"],
        default="TD3",
        help="Which agent to use.",
    )
    args = parser.parse_args(agent_args)

    # Initialize the agent based on the arguments.
    agent: Agent
    
    if args.agent == "TD3":
         agent = HockeyAgent("TD3")
    elif args.agent == "RND":
        agent = HockeyAgent("RND")
    elif args.agent == "pinkNoise":
        agent = HockeyAgent("pinkNoise")
    elif args.agent == "combi":
        agent = HockeyAgent("combi")
    else:
        raise ValueError(f"Unknown agent: {args.agent}")

    # And finally return the agent.
    return agent


def main() -> None:
    launch_client(initialize_agent)


if __name__ == "__main__":
    main()
