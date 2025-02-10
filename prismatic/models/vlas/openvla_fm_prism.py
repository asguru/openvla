import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, PreTrainedModel, AutoConfig

from prismatic.models.vlms.prismatic import PrismaticVLM

from typing import Dict, List, Optional
from PIL import Image

class OpenVLAFlowMatching(PrismaticVLM):
    def __init__(self, model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=True, arch_specifier="gelu-mlp", norm_stats: Dict[str, Dict[str, Dict[str, Dict[str, List[float]]]]]=None, **kwargs):
        super().__init__(model_id, vision_backbone, llm_backbone, enable_mixed_precision_training=enable_mixed_precision_training, arch_specifier=arch_specifier, **kwargs) 
        self.norm_stats = norm_stats

        print(self.llm_backbone.llm.config)
        self.action_expert_hidden_dim = 1024
        self.action_expert = DeepActionExpert(
            vlm_hidden_dim=self.llm_backbone.llm.config.hidden_size,
            action_expert_hidden_dim=self.action_expert_hidden_dim,
            num_heads=self.llm_backbone.llm.config.num_attention_heads,
            num_layers=(self.llm_backbone.llm.config.num_hidden_layers + 1)
        )
        action_expert_params = sum(p.numel() for p in self.action_expert.parameters())
        print("Action expert has {} params".format(action_expert_params))


    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values,
        labels,
        proprio,
        actions,
        tau,
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
            tau=tau,
            vlm_hidden_states=outputs.hidden_states,
            robot_state=proprio
        )
        return outputs, flow
    
    @torch.inference_mode()
    def predict_action(
        self, input_ids, attention_mask, pixel_values, proprio, unnorm_key: Optional[str] = None, **kwargs: str
    ) -> np.ndarray:
        """
        Core function for VLA inference; maps input image and task instruction to continuous action (de-tokenizes).

        @param image: PIL Image as [height, width, 3]
        @param instruction: Task instruction string
        @param unnorm_key: Optional dataset name for retrieving un-normalizing statistics; if None, checks that model
                           was trained only on a single dataset, and retrieves those statistics.

        @return Unnormalized (continuous) action vector --> end-effector deltas.
        """
        # image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer

        # # Build VLA Prompt
        # prompt_builder = self.get_prompt_builder()
        # prompt_builder.add_turn(role="human", message=f"What action should the robot take to {instruction.lower()}?")
        # prompt_text = prompt_builder.get_prompt()

        # # Prepare Inputs
        # input_ids = tokenizer(prompt_text, truncation=True, return_tensors="pt").input_ids.to(self.device)
        # if isinstance(tokenizer, LlamaTokenizerFast):
        #     # If the special empty token ('') does not already appear after the colon (':') token in the prompt
        #     # (after "OUT:" or "ASSISTANT:"), insert it to match the inputs seen at training time
        #     if not torch.all(input_ids[:, -1] == 29871):
        #         input_ids = torch.cat(
        #             (input_ids, torch.unsqueeze(torch.Tensor([29871]).long(), dim=0).to(input_ids.device)), dim=1
        #         )
        # else:
        #     raise ValueError(f"Unsupported `tokenizer` type = {type(tokenizer)}")

        # # Preprocess Image
        # pixel_values = image_transform(image)
        # if isinstance(pixel_values, torch.Tensor):
        #     pixel_values = pixel_values[None, ...].to(self.device)
        # elif isinstance(pixel_values, dict):
        #     pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        # else:
        #     raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            # generated_ids = super(PrismaticVLM, self).generate(
            #     input_ids=input_ids,                            # Shape: [1, seq]
            #     pixel_values=pixel_values,                      # Shape: [1, 3, res, res] or Dict[str, ...]
            #     max_new_tokens=self.get_action_dim(unnorm_key),
            #     output_hidden_states=True,
            #     # **kwargs
            # )
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                output_hidden_states=True
            )
            # fmt: on
            # Initialize torch random of shape (1, 50, 7)
            x_init = torch.randn(1, 50, 7).to(self.device)
            time_grid = torch.linspace(0,1,20).float().to(self.device)

            for (t0, t1) in zip(time_grid[:-1], time_grid[1:]):
                t0 = t0.view(1, 1, 1).expand(x_init.shape[0], 1, 1)
                t1 = t1.view(1, 1, 1).expand(x_init.shape[0], 1, 1)
                dxdt = self.action_expert(
                    action_states=x_init,
                    tau=t0,
                    vlm_hidden_states=outputs.hidden_states,
                    robot_state=proprio
                )
                dt = t1 - t0
                x_init = x_init + (dxdt * dt)

        # Un-normalize Actions
        action_norm_stats = self.get_action_stats(unnorm_key)
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = torch.tensor(action_norm_stats["q99"]).to(self.device), torch.tensor(action_norm_stats["q01"]).to(self.device)
        actions = torch.where(
            torch.tensor(mask).to(self.device),
            0.5 * (x_init + 1) * (action_high - action_low) + action_low,
            x_init,
        )

        return actions
    
    @staticmethod
    def _check_unnorm_key(norm_stats: Dict, unnorm_key: str) -> str:
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, please pass a `unnorm_key` from the following "
                f"options to choose the statistics used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        # Error Handling
        assert (
            unnorm_key in norm_stats
        ), f"The `unnorm_key` you chose is not in the set of available statistics; choose from: {norm_stats.keys()}"

        return unnorm_key

    def get_action_dim(self, unnorm_key: Optional[str] = None) -> int:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return len(self.norm_stats[unnorm_key]["action"]["q01"])

    def get_action_stats(self, unnorm_key: Optional[str] = None) -> Dict:
        """Dimensionality of the policy's action space."""
        unnorm_key = self._check_unnorm_key(self.norm_stats, unnorm_key)

        return self.norm_stats[unnorm_key]["action"]


class DeepActionExpert(nn.Module):
    def __init__(self, vlm_hidden_dim, action_expert_hidden_dim, num_heads, num_layers, max_period=10000):
        super().__init__()
        self.num_layers = num_layers
        self.vlm_hidden_dim = vlm_hidden_dim
        self.action_expert_hidden_dim = action_expert_hidden_dim
        self.max_period = max_period
        
        # Create one cross attention layer per VLM layer we want to attend to
        self.cross_attention_layers = nn.ModuleList([
            ActionCrossAttention(vlm_hidden_dim, action_expert_hidden_dim, num_heads) 
            for _ in range(num_layers)
        ])

        self.action_linear_1 = nn.Linear(7, action_expert_hidden_dim)
        self.action_linear_2 = nn.Linear(2 * action_expert_hidden_dim, action_expert_hidden_dim)
        self.action_swish = nn.SiLU()
        self.action_linear_3 = nn.Linear(action_expert_hidden_dim, action_expert_hidden_dim)

        self.proprio_linear = nn.Linear(8, action_expert_hidden_dim)
        
        # Layer norms and feed forward layers
        self.layer_norms1 = nn.ModuleList([
            nn.LayerNorm(action_expert_hidden_dim) for _ in range(num_layers)
        ])
        self.layer_norms2 = nn.ModuleList([
            nn.LayerNorm(action_expert_hidden_dim) for _ in range(num_layers)
        ])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(action_expert_hidden_dim, action_expert_hidden_dim * 4),
                nn.GELU(),
                nn.Linear(action_expert_hidden_dim * 4, action_expert_hidden_dim)
            ) for _ in range(num_layers)
        ])
        self.final_action_flow_proj = nn.Linear(action_expert_hidden_dim, 7)
        #self.tanh = nn.Tanh()

    def initial_action_transform(self, action_states, tau):
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
        half_dim = self.action_expert_hidden_dim // 2
        
        # Create the range of dimensions
        dims = torch.arange(half_dim, device=device).float()
        # Create the scale factors
        factors = torch.exp(-math.log(self.max_period) * (2 * dims / self.action_expert_hidden_dim))
        
        # Compute angles
        angles = tau * factors  # (batch_size, 1, half_dim)
        
        # Create sinusoidal encoding
        encoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        # Pad if embedding_dim is odd
        if self.action_expert_hidden_dim % 2 == 1:
            encoding = torch.cat([encoding, torch.zeros_like(encoding[...,:1])], dim=-1)
            
        return encoding.squeeze(1)  # (batch_size, embedding_dim)


    def forward(self, action_states, tau, vlm_hidden_states, robot_state, attention_mask=None):
        """
        action_states: [batch, action_seq, dim]
        tau: [batch, 1, 1]
        vlm_hidden_states: List of [batch, vlm_seq, dim] from each VLM layer
        robot_state: [batch, 1, dim]
        """
        hidden_states = self.initial_action_transform(action_states, tau)
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
        # hidden_states = self.tanh(hidden_states)
        return hidden_states

class ActionCrossAttention(nn.Module):
    def __init__(self, vlm_hidden_dim, action_expert_hidden_dim, num_heads):
        super().__init__()
        # Similar to previous implementation but with separate 
        # projections for robot state
        self.vlm_hidden_dim = vlm_hidden_dim
        self.action_expert_hidden_dim = action_expert_hidden_dim
        self.q_proj = nn.Linear(action_expert_hidden_dim, action_expert_hidden_dim)
        self.k_proj_vlm = nn.Linear(vlm_hidden_dim, action_expert_hidden_dim)
        self.v_proj_vlm = nn.Linear(vlm_hidden_dim, action_expert_hidden_dim)
        self.k_proj_robot = nn.Linear(action_expert_hidden_dim, action_expert_hidden_dim)
        self.v_proj_robot = nn.Linear(action_expert_hidden_dim, action_expert_hidden_dim)
        self.num_heads = num_heads
        self.head_dim = action_expert_hidden_dim // num_heads

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
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.action_expert_hidden_dim)
        
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
