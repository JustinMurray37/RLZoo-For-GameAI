from random import *
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from math import sqrt
from PIL import Image, ImageDraw
from copy import deepcopy

MAX_STEPS = 999999
GRAVITY = -5
JUMP = 30
BALL_SPEED = 3
BALL_RADIUS = 15
RENDER_SCALE = 0.5
AGENT_X = -185
N_NEXT_OBS = 2

def inside(point):
    """Return True if point on screen."""
    return -200 < point[0] < 200 and -200 < point[1] < 200

def abs_to_relative(balls, agent_pos):
    rel_balls = deepcopy(balls)
    for i, ball in enumerate(rel_balls):
        rel_balls[i][0] = ball[0] - agent_pos[0]
        rel_balls[i][1] = ball[1] - agent_pos[1]
    
    return rel_balls

# def sort_balls(rel_balls):
#     sorted_balls = sorted(rel_balls, key=lambda x:x[0])
#     return sorted_balls


class FlappyEnv(gym.Env):
    def __init__(self):
        super(FlappyEnv, self).__init__()
        self.render_mode = 'rgb_array'

        # Agent does not move on the x-axis
        # Max of 20 balls on x,y plain
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(low=-199, high=199, shape=(1,), dtype=np.int32),
            'balls': spaces.Box(low=-199, high=199, shape=(N_NEXT_OBS, 2,), dtype=np.int32)
        })
        # Agent can choose to "flap" or do nothing
        self.action_space = spaces.Discrete(2)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.agent = {'pos':0, 'alive':True}
        self.balls = []
        self.steps = 0

        self.state = {
            'agent': [self.agent['pos']],
            'balls': [[199,-199] for _ in range(N_NEXT_OBS)]
        }

        #simulate for 150/BALL_SPEED steps
        n_steps = 220//BALL_SPEED
        for i in range(n_steps):
            # Move each ball to the left and remove it if it goes out of bounds
            for ball in self.balls:
                ball[0] -= BALL_SPEED
            while len(self.balls) > 0 and not inside(self.balls[0]):
                self.balls.pop(0)

            if i % 10 == 0:
                y = randrange(-199, 199)
                ball = [199, y]
                self.balls.append(ball)
            
            self.state['balls'] = abs_to_relative(self.balls, [AGENT_X, self.agent['pos']])[:N_NEXT_OBS]

        return self.state, {}

    def step(self, action):
        # Move agent down, then up if the agent chooses to "flap"
        self.agent['pos'] += GRAVITY
        if action == 1:
            self.agent['pos'] += JUMP
        
        # Move each ball to the left and remove it if it goes out of bounds
        for ball in self.balls:
            ball[0] -= BALL_SPEED
        while len(self.balls) > 0 and not inside(self.balls[0]):
            self.balls.pop(0)

        if (self.steps-9) % 10 == 0:
            y = randrange(-199, 199)
            ball = [199, y]
            self.balls.append(ball)

        # If the agent is out of bounds, kill it
        if not inside([AGENT_X, self.agent['pos']]):
            self.agent['alive'] = False
        # If the agent hits a ball, kill it
        for ball in self.balls:
            diff = [ ball[0]-AGENT_X, ball[1]-self.agent['pos'] ]
            dist = sqrt(diff[0]**2 + diff[1]**2)
            if dist < BALL_RADIUS:
                self.agent['alive'] = False
        
        # Update the state
        self.state['agent'] = [self.agent['pos']]
        self.state['balls'] = abs_to_relative(self.balls, [AGENT_X, self.agent['pos']])[:N_NEXT_OBS]

        # Default reward is 1 for surviving a step
        reward = 1
        done = False
        truncated = False

        # Reward is 0 for failing
        if not self.agent['alive']:
            reward = 0
            done = True

        self.steps += 1
        if self.steps >= MAX_STEPS:
            truncated = True

        return self.state, reward, done, truncated, {}
    
    def render(self):
        image = Image.new(mode='RGB', size=(int(399*RENDER_SCALE), int(399*RENDER_SCALE)), color=(180, 180, 180))
        draw = ImageDraw.Draw(image)

        draw.circle(((AGENT_X+199)*RENDER_SCALE, -(self.agent['pos']-199)*RENDER_SCALE), 5*RENDER_SCALE, fill=(0,255,0))

        for ball in self.balls:
            draw.circle(((ball[0]+199)*RENDER_SCALE, -(ball[1]-199)*RENDER_SCALE), 10*RENDER_SCALE, fill=(0,0,0))
            
        image = np.array(image)

        return image

        
