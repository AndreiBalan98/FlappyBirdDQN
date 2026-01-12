import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import time
import argparse

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


def play_episode(env, agent, render=True, max_steps=10000):
    """
    rulează un episod cu agentul antrenat.
    
    Args:
        env: mediul
        agent: agentul DQN
        render: dacă True, randează vizual
        max_steps: steps maxime per episod
        
    Returns:
        episode_reward, episode_length
    """
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    for step in range(max_steps):
        # selectează acțiune (greedy, fără exploration)
        action = agent.select_action(state, training=False)
        
        # execută acțiune
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        state = next_state
        
        if render:
            time.sleep(0.01)  # slow down pentru vizualizare
        
        if done:
            break
    
    return episode_reward, episode_length


def evaluate_agent(model_path, n_episodes=10, render=True):
    """
    evaluează agentul pe mai multe episoade.
    
    Args:
        model_path: path la modelul antrenat
        n_episodes: număr de episoade de test
        render: dacă True, randează vizual
    """
    # creează mediu
    if render:
        env = gym.make("FlappyBird-v0", render_mode="human")
    else:
        env = gym.make("FlappyBird-v0")
    env = FlappyBirdWrapper(env)
    
    # creează și încarcă agent
    agent = DQNAgent()
    try:
        agent.load(model_path)
        print(f"model incarcat: {model_path}")
        print(f"   steps done: {agent.steps_done}")
        print(f"   epsilon: {agent.epsilon:.4f}\n")
    except FileNotFoundError:
        print(f"modelul {model_path} nu există!")
        print(f"   rulează mai întâi: python train.py")
        env.close()
        return
    
    # rulează episoade
    rewards = []
    lengths = []
    
    print(f"evaluare pe {n_episodes} episoade...\n")
    
    for episode in range(1, n_episodes + 1):
        reward, length = play_episode(env, agent, render=render)
        rewards.append(reward)
        lengths.append(length)
        
        print(f"Episode {episode:2d} | Reward: {reward:6.2f} | Length: {length:4d}")
        
        if render and episode < n_episodes:
            time.sleep(0.5)  # pauză între episoade
    
    # statistici
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    print(f"\nstatistici:")
    print(f"   reward mediu:  {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   reward max:    {max_reward:.2f}")
    print(f"   reward min:    {min_reward:.2f}")
    print(f"   length mediu:  {np.mean(lengths):.0f}")
    
    env.close()
    
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluează agent DQN pe Flappy Bird")
    parser.add_argument("--model", type=str, default="best_flappy_dqn.pth",
                        help="path la model (default: best_flappy_dqn.pth)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="număr de episoade (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="fără randare vizuală")
    
    args = parser.parse_args()
    
    # evaluare
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render
    )