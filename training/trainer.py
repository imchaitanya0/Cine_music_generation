import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

class FocalLoss(nn.Module):
    """
    Combats 'Pedal Tone Collapse' structurally by down-weighting well-classified 
    examples (like 'Sustained Ambient') and focusing gradients strictly on hard, 
    rare examples (like 'Aggressive Drive').
    """
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes] (logits)
        # targets: [batch_size] (class indices)
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # P(target class)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        else:
            return torch.sum(F_loss)

class OrchestratorTrainer:
    """
    Task 5: Advanced Mixed-Precision Master Loop
    Handles KL-Annealing, AMP Scaling, and Multi-Objective Spatial Routing
    for Kaggle T4 optimization.
    """
    def __init__(self, model, dataloader, optimizer, config, device='cuda'):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        
        # Protects Von Mises-Fisher and Exponential scaling constraints from fp16 implosion
        self.scaler = GradScaler()
        
        # Loss Initialization
        self.focal_loss_fn = FocalLoss(gamma=2.0)
        self.mse_loss_fn = nn.MSELoss()
        
        # KL-Annealing Mechanics
        self.global_step = 0
        self.kl_anneal_steps = config.get("kl_anneal_steps", 2000)
    
    def get_kl_weight(self):
        """
        Calculates mathematical beta (β) parameter for KL Divergence weight.
        Gradually warms up from 0.0 -> 1.0 to forcefully prevent Posterior Collapse.
        """
        beta = min(1.0, float(self.global_step) / float(self.kl_anneal_steps))
        return beta

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        
        loop = tqdm(self.dataloader, desc=f"Epoch {epoch}", leave=True)
        for batch_idx, batch in enumerate(loop):
            self.optimizer.zero_grad()
            self.global_step += 1
            
            # Map structural batch inputs to Hardware
            # Required keys: 'input_ids', 'attention_mask'
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Simulated continuous & categorical targets for Pipeline Validation
            tension_targets = batch.get('tension_level', torch.rand(*input_ids.shape[:2], 1)).to(self.device)
            harmony_targets = batch.get('harmony', torch.randint(0, 7, input_ids.shape[:2])).to(self.device)
            # Flatten targets mapping cleanly to scene vectors.
            
            # Launch Mixed-Precision Context Thread
            with autocast():
                # Forward Pass handles Memory Routing & Variational logic intrinsically
                outputs = self.model(input_ids, attention_mask)
                
                bs, num_scenes = outputs['categorical']['harmony'].shape[:2]
                
                # --- Multi-Objective Loss Calculation Orchestration --- #
                
                # 1. Structural CVAE Penalty
                raw_kl_loss = outputs['kl_loss']
                kl_weight = self.get_kl_weight()
                kl_penalty = kl_weight * raw_kl_loss

                # 2. Focal Losses for Categorical Imbalances
                flat_harmony_logits = outputs['categorical']['harmony'].reshape(-1, 7)
                flat_harmony_targets = harmony_targets.reshape(-1)
                
                categorical_loss = self.focal_loss_fn(flat_harmony_logits, flat_harmony_targets)
                
                # 3. Continuous Regression Constraint
                flat_tension_preds = outputs['continuous']['tension'].reshape(-1, 1)
                flat_tension_targets = tension_targets.reshape(-1, 1)
                continuous_loss = self.mse_loss_fn(flat_tension_preds, flat_tension_targets)
                
                # 4. Optional Mod-1 Riemannian SupCon (Abstracted in total pipeline loss)
                riemannian_loss = outputs.get('contrastive_loss', torch.tensor(0.0).to(self.device))
                
                # Mathematically Aggregate Complex Constraints
                loss = categorical_loss + continuous_loss + kl_penalty + (0.1 * riemannian_loss)

            # AMP Protected Backpropagation Pathway
            self.scaler.scale(loss).backward()
            
            # Safeguard Deep Mamba and Normalization architectures against Gradient Explosions
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            
            if batch_idx % 50 == 0:
                loop.set_postfix(
                    Loss=f"{loss.item():.4f}", 
                    CatFocal=f"{categorical_loss.item():.4f}", 
                    Cont=f"{continuous_loss.item():.4f}", 
                    KL=f"{raw_kl_loss.item():.4f}"
                )

        avg_loss = total_loss / len(self.dataloader)
        print(f"--- Epoch {epoch} Complete | Avg Loss: {avg_loss:.4f} ---")
        return avg_loss
