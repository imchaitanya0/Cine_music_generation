import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from peft import LoraConfig, get_peft_model
import math

class RiemannianSupConLoss(nn.Module):
    """
    Riemannian Spherical Contrastive Loss (vMF-based).
    Instead of clustering in flat Euclidean space, this mathematically constrains the 
    latent projections strictly to the surface of a hypersphere. 
    It equates temperature 'tau' to the inverse of the von Mises-Fisher concentration 
    parameter 'kappa', strictly defining geometrically equidistant spherical bounds 
    between emotion classes.
    """
    def __init__(self, temperature=0.07, device='cuda'):
        super(RiemannianSupConLoss, self).__init__()
        self.temperature = temperature # tau = 1 / kappa
        self.device = device

    def forward(self, spherical_features, labels):
        # spherical_features: [bsz, n_views, dim], must be L2 normalized across `dim`
        # Using SupCon logic operating strictly on the hypersphere
        device = (torch.device('cuda') if spherical_features.is_cuda else torch.device('cpu'))

        batch_size = spherical_features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = spherical_features.shape[1]
        contrast_feature = torch.cat(torch.unbind(spherical_features, dim=1), dim=0)
        
        # Exact Cosine Similarity (Dot product on Hypersphere)
        # anchor_dot_contrast maps exactly to the vMF exponent `kappa * cos(theta)` 
        # when divided by temperature
        anchor_dot_contrast = torch.div(
            torch.matmul(contrast_feature, contrast_feature.T),
            self.temperature)

        # Numerical stability shift
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1, 
            torch.arange(batch_size * contrast_count).view(-1, 1).to(device), 0
        )
        mask = mask.repeat(contrast_count, contrast_count) * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1.0, mask_pos_pairs)
        
        # Mean log-likelihood over positive samples
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # Final scalar loss over batch
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        return loss.view(contrast_count, batch_size).mean()


class IntensityAwarePooling(nn.Module):
    """
    Computes a contextual intensity score for each token in the sequence.
    This overrides the vanilla [CLS] token behavior by scoring how much each word
    contributes to the semantic emotion mathematically before pooling.
    """
    def __init__(self, hidden_size):
        super(IntensityAwarePooling, self).__init__()
        # 2-layer MLP to compute raw attention intensities
        self.intensity_scorer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, hidden_states, attention_mask):
        """
        hidden_states: [bsz, seq_len, hidden_size]
        attention_mask: [bsz, seq_len]
        """
        # Calculate raw intensities for every token: [bsz, seq_len, 1]
        raw_intensities = self.intensity_scorer(hidden_states)
        
        # Mask out padding tokens: set to highly negative values pre-softmax
        extended_mask = (1.0 - attention_mask.unsqueeze(-1)) * -10000.0
        masked_intensities = raw_intensities + extended_mask
        
        # Normalized token importance: [bsz, seq_len, 1]
        alpha = F.softmax(masked_intensities, dim=1)
        
        # Weighted sum of sequence over time -> [bsz, hidden_size]
        pooled_representation = torch.sum(alpha * hidden_states, dim=1)
        return pooled_representation


class RiemannianUtteranceEncoder(nn.Module):
    """
    Module 1: The Deep Mathematical Encoder.
    Applies PEFT LoRA, Intensity-Aware Pooling, and Spherical Projections.
    """
    def __init__(self, config_dict, num_emotions=7):
        super(RiemannianUtteranceEncoder, self).__init__()
        model_name = config_dict.get("model_name", "microsoft/deberta-v3-large")
        
        # Base Foundation
        self.deberta = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.deberta.config.hidden_size
        
        # Low Rank Adaptation (essential for Kaggle T4 limit)
        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            inference_mode=False,
            r=config_dict.get("lora_r", 16),      # rank
            lora_alpha=config_dict.get("lora_alpha", 32),
            lora_dropout=config_dict.get("lora_dropout", 0.1),
            target_modules=["query_proj", "value_proj", "key_proj", "dense"], 
        )
        self.deberta = get_peft_model(self.deberta, peft_config)
        
        # Advanced Custom Pooling
        self.intensity_pooler = IntensityAwarePooling(self.hidden_size)
        
        # Riemannian Projection Head (Maps to the Hypersphere space e.g. 128d)
        self.spherical_projector = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size // 2),
            nn.Linear(self.hidden_size // 2, 128)
        )
        
        self.classification_head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, num_emotions)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.deberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        
        # 1. Custom Intensity Pooling (replaces CLS Token)
        pooled_utterance = self.intensity_pooler(hidden_states, attention_mask)
        
        # 2. Spherical Projection (Constrain analytically to hypersphere ||v||_2 = 1)
        raw_projection = self.spherical_projector(pooled_utterance)
        spherical_projection = F.normalize(raw_projection, p=2, dim=1)
        
        # 3. Standard Logits
        logits = self.classification_head(pooled_utterance)
        
        return pooled_utterance, spherical_projection, logits
