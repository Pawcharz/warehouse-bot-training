
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
        
        # Embedding sizes for each modality
        visual_embedding_size = 64
        vector_embedding_size = 64
        
        # Calculate repeated vector size to match visual embedding dimension
        repeat_factor = visual_embedding_size // vector_obs_size
        actual_repeated_vector_size = repeat_factor * vector_obs_size
        
        self.repeat_factor = repeat_factor
        self.vector_net_input_size = actual_repeated_vector_size

        # Visual encoder: CNN for feature extraction
        self.visual_encoder_cnn = nn.Sequential(
            nn.Conv2d(bands, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout2d(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout2d(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
            nn.Flatten(),
        )

        # Compute CNN output size
        dummy_input = th.zeros(1, bands, visual_obs_size[1], visual_obs_size[2])
        with th.no_grad():
            visual_encoder_cnn_out_size = self.visual_encoder_cnn(dummy_input).shape[1]

        # Visual encoder: MLP for embedding
        self.visual_encoder_mlp = nn.Sequential(
            nn.Linear(visual_encoder_cnn_out_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, visual_embedding_size),
            nn.LayerNorm(visual_embedding_size)
        )
        
        # Vector encoder
        self.vector_encoder = nn.Sequential(
            nn.Linear(self.vector_net_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, vector_embedding_size),
            nn.LayerNorm(vector_embedding_size)
        )
        
        # Fusion layer input size
        fusion_size = visual_embedding_size + vector_embedding_size

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, act_dim)
        )
        
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(fusion_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        if device is not None:
            self.to(self.device)
            
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights for stable training"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _encode_observations(self, observations):
        """Shared encoding for both policy and value networks"""
        image = observations["visual"]
        vector = observations["vector"]

        # Normalize inputs
        vector = (vector - 0.5) * 2.0  # Vector to [-1, 1]
        image = image / 255            # Image to [0, 1]
        
        # Convert to tensors and move to device
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

        # Repeat vector to match embedding dimension
        repeated_vector = vector.repeat(1, self.repeat_factor)

        # Extract features from both modalities
        image_features = self.visual_encoder_cnn(image)
        image_features = self.visual_encoder_mlp(image_features)
        vector_features = self.vector_encoder(repeated_vector)

        # Fuse modalities by concatenation
        fused = th.cat([image_features, vector_features], dim=1)
        
        return fused
    
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
