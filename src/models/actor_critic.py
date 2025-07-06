import torch as th
import torch.nn as nn
from torch.distributions import Categorical

class Swish(nn.Module):
    def forward(self, x):
        return x * th.sigmoid(x)

# Actor-Critic Network for only vector observations with separate policy and value networks
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()

        print(f"obs_dim: {obs_dim}")
        
        # Separate policy network
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, act_dim)
        )
        
        # Separate value network
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.actor(x), self.critic(x)

    def get_action(self, obs, deterministic=False):
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

# Actor-Critic Network for multimodal observations with separate policy and value networks
class ActorCriticMultimodal(nn.Module):
    def __init__(self, act_dim, visual_size=[3, 36, 64], vector_obs_size=128):
        super().__init__()
        bands = visual_size[0]

        # Shapes of image and vector inputs: [<batch size>, <bands, height, width>], [<batch size>, <length>]

        visual_out_size = 64
        vector_out_size = 32

        # Visual Encoder (shared between policy and value)
        self.visual_encoder_cnn = nn.Sequential(
            nn.Conv2d(bands, 16, kernel_size=5, stride=4, padding=0),
            nn.LeakyReLU(0.01),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
        )

        # Compute flattened visual output size from dummy input
        dummy_input = th.zeros(1, bands, visual_size[1], visual_size[2])
        with th.no_grad():
            visual_encoder_cnn_out_size = self.visual_encoder_cnn(dummy_input).shape[1]

        self.visual_encoder_mlp = nn.Sequential(
            nn.Linear(visual_encoder_cnn_out_size, 64),
            Swish(),
            nn.Linear(64, visual_out_size),
            Swish()
        )
        
        # Vector Encoder (shared between policy and value)
        self.vector_encoder = nn.Sequential(
            nn.Linear(vector_obs_size, 32),
            Swish(),
            nn.Linear(32, vector_out_size),                             
            Swish()
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

    def _encode_observations(self, observations):
        """Shared encoding for both policy and value networks"""
        image = observations["image"].float()
        vector = observations["vector"]

        image_features = self.visual_encoder_cnn(image)
        image_features = self.visual_encoder_mlp(image_features)
        vector_features = self.vector_encoder(vector)

        combined = th.cat([image_features, vector_features], dim=1)
        return combined
    
    def forward(self, observations):
        combined = self._encode_observations(observations)
        return self.policy_net(combined), self.value_net(combined)

    def get_action(self, obs):
        logits, value = self.forward(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value.squeeze()

    def evaluate_actions(self, obs, actions):
        logits, values = self.forward(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, entropy, values.squeeze()

def count_parameters(model):
    """
    Count parameters in each block of the network and total parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Dictionary containing parameter counts for each block and total
    """
    total_params = 0
    block_params = {}
    
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        block_params[name] = params
        total_params += params
        
    block_params['total'] = total_params
    return block_params 