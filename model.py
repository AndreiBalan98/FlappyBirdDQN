import torch
import torch.nn as nn


class DQN(nn.Module):
    """
    CNN simplu pentru DQN pe Flappy Bird.
    Input: (batch, 4, 84, 84)
    Output: (batch, 2) - Q-values pentru cele 2 acțiuni
    """
    
    def __init__(self, n_actions=2):
        super(DQN, self).__init__()
        
        # Convolutional layers
        self.conv = nn.Sequential(
            # Conv1: 4 -> 32 channels
            nn.Conv2d(10, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            
            # Conv2: 32 -> 64 channels
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            
            # Conv3: 64 -> 64 channels
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Calculează dimensiunea output-ului conv layers
        conv_out_size = self._get_conv_output_size((10, 64, 64))
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def _get_conv_output_size(self, shape):
        """Calculează dimensiunea output-ului după conv layers"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *shape)
            output = self.conv(dummy_input)
            return int(output.numel())
    
    def forward(self, x):
        """
        Forward pass.
        x: (batch, 4, 84, 84)
        return: (batch, n_actions)
        """
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x