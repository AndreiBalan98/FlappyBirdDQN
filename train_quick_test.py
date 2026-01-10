"""
Test rapid al training loop-ului (50 episoade pentru verificare)
"""
from train import train_dqn

print("⚡ Quick test: 50 episoade pentru verificare\n")

rewards, lengths = train_dqn(
    n_episodes=50,
    max_steps_per_episode=1000,
    save_freq=25,
    log_freq=5,
    save_path="test_flappy_dqn.pth"
)

print(f"\n✅ Quick test finalizat!")
print(f"   Episoade rulate: {len(rewards)}")
print(f"   Reward mediu ultim 10: {sum(rewards[-10:])/10:.2f}")