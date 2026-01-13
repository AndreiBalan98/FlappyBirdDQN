import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
from collections import deque
import time

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


def train_dqn(
    n_episodes=5000,
    max_steps_per_episode=10000,
    save_freq=100,
    log_freq=10,
    save_path="flappy_dqn.pth"
):
    env = gym.make("FlappyBird-v0")
    env = FlappyBirdWrapper(env)
    
    agent = DQNAgent(
        n_actions=2,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=0.80,
        epsilon_end=0.00,
        epsilon_decay=100000,
        buffer_capacity=100000,
        batch_size=32,
        target_update_freq=2500
    )
    
    print(f"Training DQN pe Flappy Bird")
    print(f"   Device: {agent.device}")
    print(f"   Episoade: {n_episodes}")
    print(f"   Save frequency: {save_freq} episoade")
    print(f"   Log frequency: {log_freq} episoade\n")
    
    episode_rewards = []
    episode_lengths = []
    recent_rewards = deque(maxlen=100)
    best_avg_reward = -float('inf')
    
    start_time = time.time()
    
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        losses = []
        
        for step in range(max_steps_per_episode):
            action, _ = agent.select_action(state, training=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.store_transition(state, action, reward, next_state, done)
            
            loss = agent.train_step()
            if loss is not None:
                losses.append(loss)
            
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        recent_rewards.append(episode_reward)
        
        if episode % log_freq == 0:
            avg_reward = np.mean(recent_rewards)
            avg_loss = np.mean(losses) if losses else 0
            elapsed = time.time() - start_time
            
            print(f"Episode {episode:5d} | "
                  f"Reward: {episode_reward:6.2f} | "
                  f"Avg(100): {avg_reward:6.2f} | "
                  f"Length: {episode_length:4d} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"ε: {agent.epsilon:.3f} | "
                  f"Buffer: {len(agent.memory):6d} | "
                  f"Time: {elapsed:.0f}s")
            
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                agent.save(f"best_{save_path}")
                print(f"           Best model saved! Avg reward: {best_avg_reward:.2f}")
        
        if episode % save_freq == 0:
            agent.save(save_path)
            print(f"           Model saved at episode {episode}")
    
    agent.save(save_path)
    env.close()
    
    total_time = time.time() - start_time
    print(f"\n Training finished!")
    print(f"   Total episodes: {n_episodes}")
    print(f"   Total time: {total_time/60:.1f} minutes")
    print(f"   Best avg reward (100 ep): {best_avg_reward:.2f}")
    print(f"   Final model: {save_path}")
    print(f"   Best model: best_{save_path}")
    
    return episode_rewards, episode_lengths


if __name__ == "__main__":
    rewards, lengths = train_dqn(
        n_episodes=500,
        max_steps_per_episode=10000,
        save_freq=100,
        log_freq=10,
        save_path="flappy_dqn.pth"
    )