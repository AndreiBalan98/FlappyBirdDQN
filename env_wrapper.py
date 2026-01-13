import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image


class FlappyBirdWrapper(gym.Wrapper):
    def __init__(self, env, img_size=64, stack_frames=4, frame_skip=3):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        self.frame_skip = frame_skip
        
        self.frames = deque(maxlen=stack_frames)
        self.frame_counter = 0
        self.last_frame = None
        
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, img_size, img_size),
            dtype=np.float32
        )
    
    def _preprocess_frame(self, frame):
        img = Image.fromarray(frame).convert('L')
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        frame = np.array(img, dtype=np.float32) / 255.0
        return frame
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        frame = self._preprocess_frame(obs)
        self.last_frame = frame
        self.frame_counter = 0
        
        for _ in range(self.stack_frames):
            self.frames.append(frame)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        frame = self._preprocess_frame(obs)
        self.last_frame = frame
        
        if self.frame_counter % self.frame_skip == 0:
            self.frames.append(frame)
        
        self.frame_counter += 1
        
        return self._get_stacked_frames(), reward, terminated, truncated, info
    
    def _get_stacked_frames(self):
        return np.stack(self.frames, axis=0)