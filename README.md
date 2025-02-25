# RL Hockey Project
_by Line Abele, Meike Kriegeskorte, Pia Moeßner (nikituebi)_

- our final agents can be found under: agents/participating_in_competition (TD3_pure_final.pth = nikituebi-td3, TD3_pinkNoise_final.pth = nikituebi-pinknoise, TD3_RND_final.pth = nikituebi-rnd, TD3_combi_final.pth = nikituebi)
- Training new agents: 
  - insert hockey_env.py into Hockey environment (we added reward_goal_direction and reward_same_y to the info)
- Testing the agents:
  - in evaluation/Hockey-Env-simulation: we added the part "Test trained agent" where various test games of (trained) agent can be simulated and game data is analyzed
  - in competition directory: load .pkl files (game data of tournament) and analyze games with competition_results_plots