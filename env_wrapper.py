import gymnasium as gym
import numpy as np
from collections import deque
from PIL import Image


class FlappyBirdWrapper(gym.Wrapper):
    """
    Wrapper pentru preprocesarea input-ului Flappy Bird:
    - Conversie la grayscale
    - Resize la 84x84
    - Stack de 4 frame-uri
    - Normalizare la [0, 1]
    - Action penalty pentru jump spam
    """
    
    def __init__(self, env, img_size=84, stack_frames=4):
        super().__init__(env)
        self.img_size = img_size
        self.stack_frames = stack_frames
        
        # Deque pentru stocarea frame-urilor
        self.frames = deque(maxlen=stack_frames)
        
        # Tracking pentru action penalty
        self.last_action = 0
        self.consecutive_jumps = 0
        
        # Actualizează observation space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(stack_frames, img_size, img_size),
            dtype=np.float32
        )
    
    def _preprocess_frame(self, frame):
        """Conversie RGB -> Grayscale -> Resize -> Normalize"""
        # Conversie la grayscale
        img = Image.fromarray(frame).convert('L')
        
        # Resize
        img = img.resize((self.img_size, self.img_size), Image.BILINEAR)
        
        # Conversie la numpy și normalizare
        frame = np.array(img, dtype=np.float32) / 255.0
        
        return frame
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Reset action tracking
        self.last_action = 0
        self.consecutive_jumps = 0
        
        # Preprocesează frame-ul inițial
        frame = self._preprocess_frame(obs)
        
        # Umple stack-ul cu același frame
        for _ in range(self.stack_frames):
            self.frames.append(frame)
        
        return self._get_stacked_frames(), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Preprocesează și adaugă noul frame
        frame = self._preprocess_frame(obs)
        self.frames.append(frame)
        
        # Reward shaping pentru învățare mai bună
        shaped_reward = reward
        
        # Penalizare la moarte
        if terminated:
            shaped_reward = -1.0
        # Per frame supraviețuit
        elif reward <= 0.2:
            shaped_reward = 0.1
        # Tub trecut - bonus mare!
        else:
            shaped_reward = 1.0
        
        # ACTION PENALTY: Penalizează jump-uri consecutive excesive
        if action == 1:  # Jump
            if self.last_action == 1:
                self.consecutive_jumps += 1
                # Penalizare crescătoare pentru spam
                shaped_reward -= 0.05 * self.consecutive_jumps
            else:
                self.consecutive_jumps = 1
        else:  # Do nothing
            self.consecutive_jumps = 0
        
        # Update last action
        self.last_action = action
        
        return self._get_stacked_frames(), shaped_reward, terminated, truncated, info
    
    def _get_stacked_frames(self):
        """Returnează stack-ul de frame-uri ca array numpy"""
        return np.stack(self.frames, axis=0)