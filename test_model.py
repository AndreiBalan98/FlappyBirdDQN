import torch
from model import DQN

# Creează modelul
model = DQN(n_actions=2)
print("✓ Model creat")

# Afișează arhitectura
print("\nArhitectură:")
print(model)

# Numără parametri
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n✓ Parametri totali: {total_params:,}")
print(f"  Parametri antrenabili: {trainable_params:,}")

# Test forward pass cu batch size 1
batch_size = 1
dummy_input = torch.randn(batch_size, 4, 84, 84)
output = model(dummy_input)

print(f"\n✓ Forward pass (batch=1)")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")
print(f"  Q-values: {output.detach().numpy()}")

# Test forward pass cu batch size 32
batch_size = 32
dummy_input = torch.randn(batch_size, 4, 84, 84)
output = model(dummy_input)

print(f"\n✓ Forward pass (batch=32)")
print(f"  Input shape: {dummy_input.shape}")
print(f"  Output shape: {output.shape}")

# Verificări
assert output.shape == (batch_size, 2), f"Expected shape (32, 2), got {output.shape}"

print("\n✅ Toate verificările au trecut!")
print(f"   Model gata pentru antrenare")