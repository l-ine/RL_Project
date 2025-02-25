# RL Hockey Project
_by Line Abele, Meike Kriegeskorte, Pia Moeßner (nikituebi)_

## Overview
This project implements a Reinforcement Learning (RL) framework for training and evaluating agents in a hockey environment. It includes training scripts, agent implementations, competition settings, and evaluation tools.

## Repository Structure

```
RL_Project-main/
│-- Hockey_training/      # Scripts for training RL agents
│-- agents/               # Our final agents
│-- competition/          # Competition setup and evaluation framework
│-- evaluation/           # Scripts for evaluating trained models
│-- hockey_env.py         # Hockey environment implementation
│-- .gitignore            # Git ignore file
│-- README.md             # Project documentation
```

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/RL_Project-main.git
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Run the environment setup (if required):
   ```sh
   python hockey_env.py --setup
   ```

## Usage
 
### Training an Agent
To train an agent, navigate to the `Hockey_training/` directory and run:
```sh
python train_agent.py --agent <agent_name>
```
Insert hockey_env.py into Hockey environment (we added reward_goal_direction and reward_same_y to the info)

### Evaluating an Agent
To evaluate a trained agent, use:
```sh
python evaluation/evaluate.py --agent <agent_name>
```
In evaluation/Hockey-Env-simulation: we added the part "Test trained agent" where various test games of (trained) agent can be simulated and game data is analyzed.
In competition directory: load .pkl files (game data of tournament) and analyze games with competition_results_plots

### Competing Agents
To run a competition between agents, execute:
```sh
python competition/run_competition.py --agent1 <agent1> --agent2 <agent2>
```

### Our final Agents
1. TD3_pure_final.pth = nikituebi-td3
2. TD3_pinkNoise_final.pth = nikituebi-pinknoise
3. TD3_RND_final.pth = nikituebi-rnd
4. TD3_combi_final.pth = nikituebi

## Contributing
Feel free to fork the repository and submit pull requests. If you encounter any issues, please open an issue.

## License
This project is licensed under the MIT License.

## Acknowledgments
- Inspired by RL-based sports simulations.
- Uses OpenAI Gym environments.
