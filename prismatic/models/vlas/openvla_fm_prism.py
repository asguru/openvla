import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, PreTrainedModel, AutoConfig

from prismatic.models.vlms.prismatic import PrismaticVLM

class OpenVLAFlowMatching(PrismaticVLM):
    def __init__(self, model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=True, arch_specifier="gelu-mlp", **kwargs):
        super().__init__(model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=enable_mixed_precision_training, arch_specifier=arch_specifier, **kwargs) 

        print(self.llm_backbone.llm.config)
        self.action_expert = DeepActionExpert(
            hidden_dim=self.llm_backbone.llm.config.hidden_size,
            num_heads=self.llm_backbone.llm.config.num_attention_heads,
            num_layers=(self.llm_backbone.llm.config.num_hidden_layers + 1)
        )
        


    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        proprio,
        actions,
        output_hidden_states=True
    ):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
            output_hidden_states=output_hidden_states
        )
        flow = self.action_expert(
            action_states=actions,
            vlm_hidden_states=outputs.hidden_states,
            robot_state=proprio
        )
        return outputs, flow


class DeepActionExpert(nn.Module):
    def __init__(self, hidden_dim, num_heads, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Create one cross attention layer per VLM layer we want to attend to
        self.cross_attention_layers = nn.ModuleList([
            ActionCrossAttention(hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])

        self.action_linear_1 = nn.Linear(7, hidden_dim)
        self.action_linear_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.action_swish = nn.SiLU()
        self.action_linear_3 = nn.Linear(hidden_dim, hidden_dim)

        self.proprio_linear = nn.Linear(8, hidden_dim)
        
        # Layer norms and feed forward layers
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.final_action_flow_proj = nn.Linear(hidden_dim, 7)
        self.tanh = nn.Tanh()

    def initial_action_transform(self, action_stataes, tau):
        """
        action_states: [batch, action_seq, dim]
        tau: [batch, 1, 1]
        """
        tau_embeddings = self.sinusoidal_embedding(tau)
        action_embeddings = self.action_linear_1(action_states)
        # Concatenate the two along the last dimension. Tau is of shape (batch_size, embedding_dim),
        # and action_embeddings is of size (batch_size, action_seq, embedding_dim)
        action_tau_embeddings = torch.cat([action_embeddings, tau_embeddings.unsqueeze(1).expand(-1, action_states.shape[1], -1)], dim=-1)
        action_tau_embeddings = self.action_linear_2(action_tau_embeddings)
        action_tau_embeddings = self.action_swish(action_tau_embeddings)
        action_tau_embeddings = self.action_linear_3(action_tau_embeddings)
        return action_tau_embeddings


    def sinusoidal_embedding(self, tau):
        """
        tau: [batch, 1, 1]
        """
        device = tau.device
        half_dim = self.hidden_dim // 2
        
        # Create the range of dimensions
        dims = torch.arange(half_dim, device=device).float()
        # Create the scale factors
        factors = torch.exp(-math.log(self.max_period) * (2 * dims / self.embedding_dim))
        
        # Compute angles
        angles = tau * factors  # (batch_size, 1, half_dim)
        
        # Create sinusoidal encoding
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # Pad if embedding_dim is odd
        if self.hidden_dim % 2 == 1:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[...,:1])], dim=-1)
            
        return encoding.squeeze(1)  # (batch_size, embedding_dim)


    def forward(self, action_states, tau, vlm_hidden_states, robot_state, attention_mask=None):
        """
        action_states: [batch, action_seq, dim]
        tau: [batch, 1, 1]
        vlm_hidden_states: List of [batch, vlm_seq, dim] from each VLM layer
        robot_state: [batch, 1, dim]
        """
        hidden_states = self.initial_action_transform(action_states)
        robot_state = self.proprio_linear(robot_state)

        # Process through each layer, attending to corresponding VLM layer
        for i in range(self.num_layers):
            # Cross attention
            residual = hidden_states
            hidden_states = self.layer_norms1[i](hidden_states)
            hidden_states = self.cross_attention_layers[i](
                hidden_states,
                vlm_hidden_states[i],  # Use hidden states from corresponding VLM layer
                robot_state,
                attention_mask
            )
            hidden_states = residual + hidden_states

            # FFN
            residual = hidden_states
            hidden_states = self.layer_norms2[i](hidden_states)
            hidden_states = self.ffns[i](hidden_states)
            hidden_states = residual + hidden_states

        hidden_states = self.final_action_flow_proj(hidden_states)
        # Apply tanh to keep to -1, 1 range
        # FIXME: Look into whether this is the best choice
        hidden_states = self.tanh(hidden_states)
        return hidden_states

class ActionCrossAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # Similar to previous implementation but with separate 
        # projections for robot state
        self.hidden_dim = hidden_dim
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_vlm = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_vlm = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj_robot = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj_robot = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, action_states, vlm_states, robot_state, attention_mask=None):
        batch_size = action_states.shape[0]

        # Project queries, keys, values
        queries = self.q_proj(action_states)
        
        # Separate projections for VLM and robot state
        keys_vlm = self.k_proj_vlm(vlm_states)
        values_vlm = self.v_proj_vlm(vlm_states)
        keys_robot = self.k_proj_robot(robot_state)
        values_robot = self.v_proj_robot(robot_state)
        
        # Concatenate keys and values
        keys = torch.cat([keys_vlm, keys_robot], dim=1)
        values = torch.cat([values_vlm, values_robot], dim=1)

        # Reshape for multi-head attention
        # [batch, heads, seq, head_dim]
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [batch, heads, action_seq, context_seq]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax over context sequence dimension
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Get output values
        # [batch, heads, action_seq, head_dim]
        output = torch.matmul(attention_probs, values)

        # Reshape back
        # [batch, action_seq, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        return output

# Usage example:
# Get all hidden states from VLM
# vlm_all_hidden_states = output.hidden_states  # List of tensor for each layer

# action_expert = DeepActionExpert(
#     hidden_dim=768, 
#     num_heads=12,
#     num_layers=len(vlm_all_hidden_states)
# )

# output = action_expert(
#     action_states=action_embeddings,
#     vlm_hidden_states=vlm_all_hidden_states,
#     robot_state=robot_state_embedding,
#     attention_mask=attention_mask
# )

class ActionExpert(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        # Projection layers for queries, keys, values
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def forward(self, action_states, vlm_hidden_states, robot_state, attention_mask=None):
        batch_size = action_states.shape[0]
        
        # Concatenate all previous states that actions can attend to
        # [batch, seq, dim]
        context = torch.cat([vlm_hidden_states, robot_state], dim=1)
        
        # Project queries from action states
        # [batch, action_seq, dim]
        queries = self.q_proj(action_states)
        
        # Project keys/values from context
        # [batch, context_seq, dim] 
        keys = self.k_proj(context)
        values = self.v_proj(context)

        # Reshape for multi-head attention
        # [batch, heads, seq, head_dim]
        queries = queries.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        # [batch, heads, action_seq, context_seq]
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Add attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax over context sequence dimension
        attention_probs = F.softmax(attention_scores, dim=-1)

        # Get output values
        # [batch, heads, action_seq, head_dim]
        output = torch.matmul(attention_probs, values)

        # Reshape back
        # [batch, action_seq, dim]
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)

        return output

# Example usage:
# vlm_states = output.hidden_states[-1]  # Get last layer hidden states from VLM
# action_states = ... # Your action token embeddings
# robot_state = ... # Robot state embeddings

# Create attention mask that allows actions to attend to all VLM tokens
# and robot state but not to other action tokens
# seq_len = vlm_states.shape[1] + robot_state.shape[1] + action_states.shape[1]
# attention_mask = torch.zeros((seq_len, seq_len))
# action_start = vlm_states.shape[1] + robot_state.shape[1]
# attention_mask[action_start:, action_start:] = float('-inf')  # Mask future action tokens

# action_expert = ActionExpert(hidden_dim=768, num_heads=12)
# output = action_expert(action_states, vlm_states, robot_state, attention_mask)
