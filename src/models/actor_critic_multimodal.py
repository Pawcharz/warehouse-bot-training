
import torch as th
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np

# Actor-Critic Network for multimodal observations with separate policy and value networks
class ActorCriticMultimodal(nn.Module):
    def __init__(self, act_dim, visual_obs_size, vector_obs_size, device=None):
        super().__init__()
        self.device = device

        bands = visual_obs_size[0]
        # Shapes of image and vector inputs: [<batch size>, <bands, height, width>], [<batch size>, <length>]

        visual_out_size = 32
        vector_out_size = 2

        # Visual Encoder (shared between policy and value)
        self.visual_encoder_cnn = nn.Sequential(
            nn.Conv2d(bands, 6, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )

        # Compute flattened visual output size from dummy input
        dummy_input = th.zeros(1, bands, visual_obs_size[1], visual_obs_size[2])
        with th.no_grad():
            visual_encoder_cnn_out_size = self.visual_encoder_cnn(dummy_input).shape[1]

        self.visual_encoder_mlp = nn.Sequential(
            nn.Linear(visual_encoder_cnn_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, visual_out_size),
            nn.Tanh()
        )
        
        # Vector Encoder (shared between policy and value)
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_obs_size, vector_out_size),
            nn.Tanh(),
        )

        # Separate Policy Network
        self.policy_net = nn.Sequential(
            nn.Linear(visual_out_size + vector_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, act_dim)
        )
        
        # Separate Value Network
        self.value_net = nn.Sequential(
            nn.Linear(visual_out_size + vector_out_size, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Move model to device if specified
        if device is not None:
            self.to(self.device)

    def _encode_observations(self, observations):
        """Shared encoding for both policy and value networks"""

        image = observations["visual"]
        vector = observations["vector"]
        
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

        combined = th.cat([image_features, vector_features], dim=1)
        return combined
    
    def forward(self, observations):
        combined = self._encode_observations(observations)
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
