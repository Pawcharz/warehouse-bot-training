
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# Actor-Critic Network for multimodal observations with separate policy and value networks
class ActorCriticMultimodal(nn.Module):
    def __init__(self, act_dim, visual_obs_size, vector_obs_size, device=None):
        super().__init__()
        self.device = device
        
        # Store latest gate coefficient for logging
        self.latest_gate_coeff = None
        # Store current gate tensor for loss calculation
        self.current_gate = None

        bands = visual_obs_size[0]
        # Shapes of image and vector inputs: [<batch size>, <bands, height, width>], [<batch size>, <length>]

        # Modality out size is the output size of the single modality encoder (modalities are added together)
        modality_out_size = 128

        self.visual_encoder_cnn = nn.Sequential(
            # First conv block
            nn.Conv2d(bands, 16, kernel_size=5, stride=1, padding=2),  # Increased filters, added padding
            nn.BatchNorm2d(16),
            nn.ReLU(),  # Changed from Tanh to ReLU
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # More filters
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth conv block for better feature extraction
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),  # Adaptive pooling for consistent output
            nn.Flatten(),
            nn.Dropout(0.1)  # Light dropout for regularization
        )

        # Compute flattened visual output size from dummy input
        dummy_input = th.zeros(1, bands, visual_obs_size[1], visual_obs_size[2])
        with th.no_grad():
            visual_encoder_cnn_out_size = self.visual_encoder_cnn(dummy_input).shape[1]

        self.visual_encoder_mlp = nn.Sequential(
            nn.Linear(visual_encoder_cnn_out_size, 256),  # Increased capacity
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, modality_out_size),
            nn.LayerNorm(modality_out_size)
        )
        
        # Vector Encoder (shared between policy and value)
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_obs_size, 32),
            nn.ReLU(),
            nn.Linear(32, modality_out_size),
            nn.LayerNorm(modality_out_size)
        )

        # Gate is computed from concatenated features
        self.gate_fc = nn.Sequential(
            nn.Linear(modality_out_size + modality_out_size, 1),
            nn.Sigmoid()
        )
        
        # Fusion size is the output size of the added features
        fusion_size = modality_out_size

        # Separate Policy Network
        self.policy_net = nn.Sequential(
            nn.Linear(fusion_size, 128),     # Much larger
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        
        # Separate Value Network
        self.value_net = nn.Sequential(
            nn.Linear(fusion_size, 128),     # Much larger
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # Move model to device if specified
        if device is not None:
            self.to(self.device)

    def _encode_observations(self, observations):
        """Shared encoding for both policy and value networks"""

        image = observations["visual"]
        vector = observations["vector"]

        # Normalize vector inputs to [-1, 1] range
        vector = (vector - 0.5) * 2.0

        # Normalize image inputs to [0, 1] range assuming it originally was in [0, 255]
        image = image / 255
        
        # Convert to tensors and move to model device
        if not isinstance(image, th.Tensor):
            image = th.tensor(image, device=self.device)
        else:
            image = image.to(self.device)
            
        if not isinstance(vector, th.Tensor):
            vector = th.tensor(vector, device=self.device)
        else:
            vector = vector.to(self.device)
            
        image = image.float()
        vector = vector.float()

        # Encode features
        image_features = self.visual_encoder_cnn(image)
        image_features = self.visual_encoder_mlp(image_features)
        vector_features = self.vector_encoder(vector)

        # Compute gating weight (scalar between 0 and 1)
        concat = th.cat([image_features, vector_features], dim=1)  # [B, image_dim + vector_dim]
        gate = self.gate_fc(concat)  # [B, 1]
        
        # Store gate coefficient for logging (detach to avoid affecting gradients)
        self.latest_gate_coeff = gate.detach().mean().item()

        # Fuse using learned gate
        fused = gate * image_features + (1 - gate) * vector_features  # [B, fusion_dim]

        # Fusion of image and vector features as addition of the two features (with gating)
        # fused = image_features + vector_features
        return fused, gate  # Return both fused features and gate tensor
    
    def forward(self, observations):
        combined, gate = self._encode_observations(observations)
        # Store gate tensor for loss calculation (keep gradients)
        self.current_gate = gate
        return self.policy_net(combined), self.value_net(combined)

    def get_action(self, obs, deterministic=False):
        """Get action from observations"""
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        if deterministic:
            action = dist.probs.argmax(dim=-1)
        else:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze()

    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze()
    
    def get_latest_gate_coeff(self):
        """Get the latest gate coefficient for logging"""
        return self.latest_gate_coeff
    
    def get_current_gate(self):
        """Get the current gate tensor for loss calculation (preserves gradients)"""
        return getattr(self, 'current_gate', None)
