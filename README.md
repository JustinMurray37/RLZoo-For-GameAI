# Comparison of RL Algorithms
## Comparison of DQN, PPO, A2C, and TRPO in a Flappy Bird game clone. Implemented using Gymnasium and SB3.
Changes since my original implementation of a Flappy Bird like game:
* Agent is placed at the far left
* Obstacles are spawned every 10 steps instead of a 10% chance at each step
* Upon creation/reset of the environment, it is simulated for a fixed number of steps to populate most of the area with obstacles before the start of the episode
* Input now uses the relative position of the next two obstacles, rather than the absolute position of all obstacles

[WandB Report](https://wandb.ai/jm104018-ou/rl-zoo-with-flappy/reports/Comparison-of-RL-Algorithms--VmlldzoxNjQwNTM4Ng)
