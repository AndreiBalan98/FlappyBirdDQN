import gymnasium as gym
import flappy_bird_gymnasium
import numpy as np
import time
import argparse
import cv2

from env_wrapper import FlappyBirdWrapper
from dqn_agent import DQNAgent


def play_episode(env, agent, render=True, max_steps=10000, show_qvalues=True, show_frame=True):
    state, _ = env.reset()
    episode_reward = 0
    episode_length = 0
    
    if show_frame:
        cv2.namedWindow('Network Input (64x64 Grayscale)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Network Input (64x64 Grayscale)', 256, 256)
    
    for step in range(max_steps):
        action, q_values = agent.select_action(state, training=False)
        
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        episode_reward += reward
        episode_length += 1
        
        if show_qvalues:
            print(f"\rStep {step:4d}   Q[wait]={q_values[0]:7.3f}   Q[jump]={q_values[1]:7.3f}   Action: {'JUMP' if action == 1 else 'WAIT'}   Frame reward: {reward:+6.2f}   Total: {episode_reward:7.2f}   ", end='', flush=True)
        
        if show_frame:
            latest_frame = state[-1]
            frame_display = (latest_frame * 255).astype(np.uint8)
            cv2.imshow('Network Input (64x64 Grayscale)', frame_display)
            cv2.waitKey(1)
        
        state = next_state
        
        if render:
            time.sleep(0.01)
        
        if done:
            if show_qvalues:
                print()
            break
    
    if show_frame:
        cv2.destroyAllWindows()
    
    return episode_reward, episode_length


def evaluate_agent(model_path, n_episodes=10, render=True, show_qvalues=True, show_frame=True):
    if render:
        env = gym.make("FlappyBird-v0", render_mode="human")
    else:
        env = gym.make("FlappyBird-v0")
    env = FlappyBirdWrapper(env)
    
    agent = DQNAgent()
    try:
        agent.load(model_path)
        print(f"model incarcat: {model_path}")
        print(f"   steps done: {agent.steps_done}")
        print(f"   epsilon: {agent.epsilon:.4f}\n")
    except FileNotFoundError:
        print(f"modelul {model_path} nu exista!")
        print(f"   ruleaza mai intai: python train.py")
        env.close()
        return
    
    rewards = []
    lengths = []
    
    print(f"evaluare pe {n_episodes} episoade...\n")
    
    for episode in range(1, n_episodes + 1):
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{n_episodes}")
        print('='*60)
        
        reward, length = play_episode(env, agent, render=render, show_qvalues=show_qvalues, show_frame=show_frame)
        rewards.append(reward)
        lengths.append(length)
        
        print(f"\nEpisode {episode:2d} | Reward: {reward:6.2f} | Length: {length:4d}")
        
        if render and episode < n_episodes:
            time.sleep(0.5)
    
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    max_reward = np.max(rewards)
    min_reward = np.min(rewards)
    
    print(f"\n{'='*60}")
    print(f"statistici:")
    print(f"   reward mediu:  {avg_reward:.2f} ± {std_reward:.2f}")
    print(f"   reward max:    {max_reward:.2f}")
    print(f"   reward min:    {min_reward:.2f}")
    print(f"   length mediu:  {np.mean(lengths):.0f}")
    print('='*60)
    
    env.close()
    
    return rewards, lengths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evalueaza agent DQN pe Flappy Bird")
    parser.add_argument("--model", type=str, default="best_flappy_dqn.pth",
                        help="path la model (default: best_flappy_dqn.pth)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="numar de episoade (default: 10)")
    parser.add_argument("--no-render", action="store_true",
                        help="fara randare vizuala")
    parser.add_argument("--no-qvalues", action="store_true",
                        help="fara afisare Q-values")
    parser.add_argument("--no-frame", action="store_true",
                        help="fara afisare frame procesat")
    
    args = parser.parse_args()
    
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        render=not args.no_render,
        show_qvalues=not args.no_qvalues,
        show_frame=not args.no_frame
    )