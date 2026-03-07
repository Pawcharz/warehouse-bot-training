
import torch as th
import torch.nn as nn
from torch.distributions import Categorical

class TaskEncoder(nn.Module):
    def __init__(self, num_items, item_embedding_dim, output_dim):
        super().__init__()
        
        # possible item states: [None, item1, item2, ...]
        # Embedding table: each item index -> embedding vector
        self.num_items = num_items
        self.item_embedding = nn.Embedding(num_items+1, item_embedding_dim) # +1 for "None" state
        self.item_embedding_dim = item_embedding_dim
        
        # Task embedding size is 2 * item_embedding_dim (pick + held item embeddings)
        task_embedding_size = 2 * item_embedding_dim
        
        # Task encoder: processes the concatenated item embeddings
        self.encoder = nn.Sequential(
            nn.Linear(task_embedding_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim)
        )

    def forward(self, task_vector):
        # vector should contain item indices [pick_item_idx, held_item_idx]
        task_vector = task_vector.int()

        pick_emb = self.item_embedding(task_vector[:, 0])
        held_emb = self.item_embedding(task_vector[:, 1])
        
        task_emb = th.cat([pick_emb, held_emb], dim=-1)  # Shape: (batch_size, 2 * item_embedding_dim)
        
        # Process through encoder network
        encoded_task = self.encoder(task_emb)  # Shape: (batch_size, output_dim)
        return encoded_task


# Actor-Critic Network for multimodal observations with configurable architecture
class ActorCriticMultimodalConfigurable(nn.Module):
    def __init__(self, act_dim, visual_obs_size, num_items, 
                 visual_encoder_blocks=4, task_embedding_dim=32, fusion_type="concat",
                 device=None):
        """
        Configurable Actor-Critic for ablation studies.
        
        Args:
            act_dim: Number of actions
            visual_obs_size: Shape of visual observation (channels, height, width)
            num_items: Number of items in the task
            visual_encoder_blocks: Number of CNN blocks (3, 4, or 5)
            task_embedding_dim: Dimension of task embedding (16, 32, or 64)
            fusion_type: How to fuse modalities (currently only "concat" supported)
            device: Device to use
        """
        super().__init__()
        self.device = device
        self.visual_encoder_blocks = visual_encoder_blocks
        self.task_embedding_dim = task_embedding_dim
        self.fusion_type = fusion_type

        bands = visual_obs_size[0]
        
        # Visual encoder: CNN for feature extraction with configurable depth
        visual_embedding_size = 64
        self.visual_encoder_cnn = self._build_visual_encoder(bands, visual_encoder_blocks)

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
        
        # Task encoder: converts item indices to embeddings and extracts features
        self.task_encoder = TaskEncoder(
            num_items=num_items,
            item_embedding_dim=task_embedding_dim,
            output_dim=visual_embedding_size
        )
        
        # Fusion layer input size
        if fusion_type == "concat":
            fusion_size = visual_embedding_size + visual_embedding_size
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        self.fusion_size = fusion_size

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

    def _build_visual_encoder(self, bands, num_blocks):
        """Build CNN encoder with configurable depth."""
        layers = []
        
        # Block 1: Always present (bands -> 16)
        layers.extend([
            nn.Conv2d(bands, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Block 2: Always present (16 -> 32)
        layers.extend([
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])
        
        # Block 3: Present for 3, 4, 5 blocks (32 -> 64)
        if num_blocks >= 3:
            layers.extend([
                nn.Dropout2d(0.1),
                nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
        
        # Block 4: Present for 4, 5 blocks (64 -> 128)
        if num_blocks >= 4:
            layers.extend([
                nn.Dropout2d(0.1),
                nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            ])
        
        # Block 5: Present for 5 blocks (128 -> 256)
        if num_blocks >= 5:
            layers.extend([
                nn.Dropout2d(0.1),
                nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
            ])
        
        # Add flatten at the end
        layers.append(nn.Flatten())
        
        # Adjust final pooling for 3-block case
        if num_blocks == 3:
            # For 3 blocks, we need adaptive pooling before flatten
            layers.insert(-1, nn.AdaptiveAvgPool2d((2, 2)))
        
        return nn.Sequential(*layers)

    def _encode_observations(self, observations):
        """Shared encoding for both policy and value networks"""
        image = observations["visual"]
        vector = observations["vector"]

        # Normalize image input
        image = image / 255.0  # Image to [0, 1]
        
        # Convert to tensors and move to device
        if isinstance(image, th.Tensor):
            image = image.to(self.device)
        else:
            image = th.tensor(image, device=self.device)
            
        if isinstance(vector, th.Tensor):
            vector = vector.to(self.device)
        else:
            vector = th.tensor(vector, device=self.device)
            
        image = image.float()
        
        # Extract features from both modalities

        # Image
        image_features = self.visual_encoder_cnn(image)
        image_features = self.visual_encoder_mlp(image_features)

        # Items/Tasks
        task_features = self.task_encoder(vector)

        # Fusion
        if self.fusion_type == "concat":
            fused = th.cat([image_features, task_features], dim=1)
        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")
        
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
