# RL Hockey Project
_by Line Abele, Meike Kriegeskorte, Pia Moeßner (nikituebi)_

## Repository Structure

```
RL_Project-main/
│-- Hockey_training/      # Scripts for training RL agents
│-- agents/               # Our final agents
│-- competition/          # Competition setup and evaluation framework
│-- evaluation/           # Scripts for evaluating trained models
│-- hockey_env.py         # Modifications of hockey_env.py in hockey package
```

## Usage

### Setup
Install requirements. The hockey_env.py provided here can be used as an alternative to the hockey_env.py of the installed hockey package (some minor changes like reward shaping).
Some scripts contain absolute paths to the Hockey_training module, these have to be adjusted.
 
### Training an Agent
To train an agent, navigate to the `Hockey_training/` directory and run:
```sh
python training.py -h 
```
- -h to see all other args. 
- In training.py checkpoints to .pth files can be alternated depending on which agent you want to use for training or playing a game.
- Execution.py to run various trainings with different parameters.

### Evaluating an Agent
- evaluation/plots_losses_rewards: evaluate if training is going well.
- evaluation/Hockey-Env-simulation: we added the part "Test trained agent" where various test games of (trained) agent can be simulated and game data is analyzed.
- competition/competition_results_plots: load .pkl files (game data of tournament) and analyze games against unknown agents.

### Our final Agents
1. TD3_pure_final.pth = nikituebi-td3
2. TD3_pinkNoise_final.pth = nikituebi-pinknoise
3. TD3_RND_final.pth = nikituebi-rnd
4. TD3_combi_final.pth = nikituebi

<br>
We want to thank Georg Martius and his lab for setting up the hockey environment and the tournament server and organizing everything around it. Playing with our agents was fun!