import my_flappy
import wandb
from stable_baselines3 import A2C, PPO, DQN
from sb3_contrib import TRPO
from gymnasium.wrappers import RecordVideo
from wandb.integration.sb3 import WandbCallback
import glob
import os
os.environ["WANDB_MODE"] = "offline"

run = wandb.init(project="rl-zoo-with-flappy", name="TRPO",
                 sync_tensorboard=True, monitor_gym=True)

env = my_flappy.FlappyEnv()
env = RecordVideo(env, video_folder=f'videos/{run.id}',
                  episode_trigger= lambda x: x%100==0)

# model = DQN('MultiInputPolicy', env, verbose=1, tensorboard_log="./tb_logs")
# model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./tb_logs")
# model = A2C('MultiInputPolicy', env, verbose=1, tensorboard_log="./tb_logs")
# model = TRPO('MultiInputPolicy', env, verbose=1, tensorboard_log="./tb_logs")

model.learn(
    total_timesteps=500000,
    callback=WandbCallback(
        gradient_save_freq=1000,
        model_save_path="./models/",
        verbose=2,
    )
)

videos = glob.glob(os.path.join(f'./videos/{run.id}', '*.mp4'))
videos.sort()
wandb.log({'media': wandb.Video(videos[-1], format='mp4')})