import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GatedChronologicalScenePooler(nn.Module):
    """
    Module 2: Gated Cross-Speaker Scene Pooling.
    Eliminates generic Multi-Head Attention for explicit conversational modeling.
    Integrates pre-softmax Exponential Chronological Decay and Gated Fill-Suppression.
    """
    def __init__(self, hidden_size):
        super(GatedChronologicalScenePooler, self).__init__()
        self.hidden_size = hidden_size
        
        # Gating Mechanism (Suppresses filler/unimportant utterances)
        self.suppression_gate = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
            nn.Sigmoid() # Outputs continuous value [0, 1] per utterance
        )
        
        # Learnable decay parameter lambda for Chronological Exponential Decay
        # Initialized to a small positive value (e.g. ln(1.05))
        self.decay_lambda = nn.Parameter(torch.tensor(0.05))
        
        # Attention scoring for pooling
        self.attention_query = nn.Parameter(torch.randn(1, 1, hidden_size))
        self.key_layer = nn.Linear(hidden_size, hidden_size)

    def forward(self, utterance_embeddings, attention_mask=None):
        """
        utterance_embeddings: [batch_size, num_utterances, hidden_size]
        attention_mask: [batch_size, num_utterances] boolean mask (True for padding)
        Returns:
            scene_vector: [batch_size, hidden_size]
        """
        bsz, seq_len, dim = utterance_embeddings.shape
        device = utterance_embeddings.device
        
        # 1. Gated Suppression Filter: g = \sigma(W_g * x)
        # Suppress features before computing importance: [bsz, seq_len, 1]
        gate_values = self.suppression_gate(utterance_embeddings)
        gated_utterances = utterance_embeddings * gate_values
        
        # 2. Compute Raw Attention Logits using the Query
        query = self.attention_query.expand(bsz, -1, -1) # [bsz, 1, dim]
        keys = self.key_layer(gated_utterances)          # [bsz, seq_len, dim]
        
        # Contextual logits scaling: [bsz, seq_len]
        logits = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(dim)
        
        # 3. Explicit Chronological Exponential Decay
        # Utterances at the end of the scene (t closer to seq_len) have less decay.
        # Penalty = -\lambda * (seq_len - t). Added before softmax.
        t_indices = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(0)
        # Max index is roughly sequence length, we use valid length from mask ideally,
        # but for vectorized penalty, valid max length is computed dynamically:
        if attention_mask is not None:
            # Number of valid tokens per batch: [bsz, 1]
            valid_lengths = (~attention_mask).sum(dim=1, keepdim=True).float()
            # Penalty is zero at the last valid token
            penalty_indices = torch.clamp(valid_lengths - 1 - t_indices, min=0.0)
        else:
            penalty_indices = seq_len - 1 - t_indices
        
        # We enforce lambda to be positive using torch.abs (or softplus)
        pos_lambda = F.softplus(self.decay_lambda)
        chronological_penalty = -pos_lambda * penalty_indices
        
        # Inject penalty into logits
        decayed_logits = logits + chronological_penalty
        
        # Mask out strict padding entirely (-Inf)
        if attention_mask is not None:
            decayed_logits = decayed_logits.masked_fill(attention_mask, -10000.0)
            
        # 4. Softmax and Pool
        alpha = F.softmax(decayed_logits, dim=1).unsqueeze(-1) # [bsz, seq_len, 1]
        
        # Final Scene Vector
        scene_vector = torch.sum(alpha * gated_utterances, dim=1)
        
        return scene_vector

from transformers import MambaConfig, MambaModel

class DifferentiableEpisodicMemory(nn.Module):
    """
    Module 3 Part A: Differentiable Episodic Memory Matrix (\mathcal{M}).
    Maintains a discrete topological space of re-usable classical narrative elements.
    Operates akin to a Differentiable Neural Computer to augment Mamba's continuous state.
    """
    def __init__(self, num_slots, dim):
        super(DifferentiableEpisodicMemory, self).__init__()
        self.num_slots = num_slots
        self.dim = dim
        
        # Explicit Matrix Memory \mathcal{M}
        self.memory = nn.Parameter(torch.Tensor(num_slots, dim))
        nn.init.orthogonal_(self.memory) # Mathematically independent slots
        
        # Read/Write routing projections
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        
        # Neural Fusion Gate
        self.fusion_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )

    def forward(self, mamba_states):
        """
        Retrieves discrete motifs by querying the memory matrix, then fuses 
        it mathematically with the continuous sequence representations.
        mamba_states: [bsz, seq_len, dim]
        """
        bsz, seq_len, _ = mamba_states.shape
        
        # 1. Generate Queries from continuous states: [bsz, seq_len, dim]
        Q = self.query_proj(mamba_states)
        
        # 2. Extract Keys/Values from discrete memory: [num_slots, dim]
        K = self.key_proj(self.memory)
        V = self.value_proj(self.memory)
        
        # 3. Explicit Cosine-Similarity Read Operation
        # Normalize for pure Cosine representation (bounds [-1, 1])
        Q_norm = F.normalize(Q, p=2, dim=-1)
        K_norm = F.normalize(K, p=2, dim=-1)
        
        # Scaled routing dot product: [bsz, seq_len, num_slots]
        # Using a parameterized temperature could follow vMF logic, here we use fixed sqrt.
        routing_scores = torch.matmul(Q_norm, K_norm.transpose(0,1)) / math.sqrt(self.dim)
        routing_weights = F.softmax(routing_scores, dim=-1)
        
        # Read state R: [bsz, seq_len, dim]
        R = torch.matmul(routing_weights, V)
        
        # 4. Deterministic Neural Routing (Fusion Gate)
        # Calculates exactly how much the discrete memory overrides the continuous state
        # F = \sigma( W * [Mamba, Read] )
        fusion_weights = self.fusion_gate(torch.cat([mamba_states, R], dim=-1))
        
        # Final interpolated state
        augmented_states = fusion_weights * mamba_states + (1.0 - fusion_weights) * R
        return augmented_states

class SceneNarrativeEngine(nn.Module):
    """
    Total Module 2 & 3 integration engine.
    Runs continuous sequences through Mathematical Gated Scene Pooling,
    tracks them via Causal Mamba-2, and grounds them via Differentiable Episodic Memory.
    """
    def __init__(self, config_dict, m1_hidden_size):
        super(SceneNarrativeEngine, self).__init__()
        
        mamba_d_model = config_dict.get("mamba_d_model", 256)
        mamba_n_layer = config_dict.get("mamba_n_layer", 4)
        
        # Mod 2: The Scene Pooler
        self.scene_pooler = GatedChronologicalScenePooler(hidden_size=m1_hidden_size)
        
        # Dimensionality Bridge
        self.input_projection = nn.Linear(m1_hidden_size, mamba_d_model)
        
        # Mod 3: Continuous State Tracking (Mamba)
        configuration = MambaConfig(
            hidden_size=mamba_d_model,
            num_hidden_layers=mamba_n_layer,
            vocab_size=1, pad_token_id=0
        )
        self.mamba = MambaModel(configuration)
        
        # Mod 3: Discrete Structural Memory Bank (Differentiable Neural Computer)
        self.episodic_memory = DifferentiableEpisodicMemory(
            num_slots=config_dict.get("num_memory_slots", 50),
            dim=mamba_d_model
        )
        
    def forward(self, batched_utterances, utterances_padding_mask):
        bs, num_scenes, num_utts, dim = batched_utterances.shape
        
        # Flatten batch and scenes to process Utterance Attention
        flat_utterances = batched_utterances.view(bs * num_scenes, num_utts, dim)
        flat_mask = utterances_padding_mask.view(bs * num_scenes, num_utts)
        
        # [A] Phase 1: Mod 2 Pooling -> [bs * num_scenes, dim]
        scene_vectors = self.scene_pooler(flat_utterances, attention_mask=flat_mask)
        scene_sequence = scene_vectors.view(bs, num_scenes, dim)
        
        # Project to Narrative engine manifold limits
        projected_scenes = self.input_projection(scene_sequence)
        
        # [B] Phase 2: Mod 3 Continuous Mamba Tracking 
        # Causal linear-time propagation: [bs, num_scenes, mamba_dim]
        mamba_outputs = self.mamba(inputs_embeds=projected_scenes).last_hidden_state
        
        # [C] Phase 3: Mod 3 Differentiable Discrete Augmentation
        augmented_narrative = self.episodic_memory(mamba_outputs)
        
        return augmented_narrative
