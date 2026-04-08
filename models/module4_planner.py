import torch
import torch.nn as nn
import torch.nn.functional as F

class KLDivergencePenalty(nn.Module):
    """
    Mathematical computation of Analytical KL Divergence.
    Calculates D_{KL}[N(\mu, \sigma^2) || N(0, 1)] analytically.
    Acts as the perfect Information Bottleneck, preventing the posterior from detaching
    from the prior, thereby resolving 'Pedal Tone Collapse' structurally.
    """
    def __init__(self):
        super(KLDivergencePenalty, self).__init__()

    def forward(self, mu, logvar):
        # Resulting loss is averaged across the batch dimension but summed dynamically 
        # across the latent dimension.
        # Formula: -0.5 * sum(1 + log_var - mu^2 - e^{log_var})
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()

class VariationalBottleneck(nn.Module):
    """
    Conditional Variational Autoencoder (CVAE) Reparameterization Space.
    Converts deterministic narrative vectors into parameterized probability distributions.
    """
    def __init__(self, input_dim, hidden_dim, latent_dim=128):
        super(VariationalBottleneck, self).__init__()
        
        self.encoder_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Probabilistic Projections
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def reparameterize(self, mu, logvar):
        """
        Samples z = \mu + \epsilon * \sigma
        Where \epsilon ~ N(0, 1)
        """
        if self.training:
            # \sigma = \exp(0.5 * \log \sigma^2)
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
            return z
        else:
            # During inference, we can drop the variance for deterministic output,
            # or sample multiple times for variational beam searching.
            return mu

    def forward(self, narrative_state):
        h = self.encoder_mlp(narrative_state)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # Constrain numerical range of logvar to prevent overflow errors
        logvar = torch.clamp(logvar, min=-20, max=5)
        
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class CVAEMusicPlanner(nn.Module):
    """
    Module 4: Deep Mathematical Variational Music Planner.
    Predicts categorical and continuous musical parameters driven by sampling
    from the latent variational distribution rather than a flat deterministic array.
    """
    def __init__(self, config_dict, mamba_d_model):
        super(CVAEMusicPlanner, self).__init__()
        
        hidden_dim = config_dict.get("hidden_dim", 512)
        latent_dim = 128
        
        self.variational_core = VariationalBottleneck(
            input_dim=mamba_d_model,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim
        )
        
        self.kl_calculator = KLDivergencePenalty()
        
        # Decoding branches (from latent z -> music)
        # We explicitly model non-deterministic structures
        
        # Categorical Heads
        self.head_tempo = nn.Linear(latent_dim, 5)
        self.head_harmony = nn.Linear(latent_dim, 7)
        self.head_texture = nn.Linear(latent_dim, 4)
        self.head_instrumentation = nn.Linear(latent_dim, 10)
        self.head_rhythmic_density = nn.Linear(latent_dim, 5)
        
        # Continuous Heads (e.g. Volume/Tension mapped [0, 1])
        self.head_tension_level = nn.Linear(latent_dim, 1)
        self.head_dynamic_swell = nn.Linear(latent_dim, 1)
        
    def forward(self, narrative_states):
        """
        narrative_states: [batch_size, num_scenes, mamba_d_model]
        """
        bs, num_scenes, dim = narrative_states.shape
        flat_states = narrative_states.reshape(-1, dim)
        
        # Embed functionally into the structural probability space
        z, mu, logvar = self.variational_core(flat_states)
        
        # Analytically calculate structural penalty
        kl_divergence = self.kl_calculator(mu, logvar)
        
        # Predict from Latent Sample (z)
        logits_tempo = self.head_tempo(z)
        logits_harmony = self.head_harmony(z)
        logits_texture = self.head_texture(z)
        logits_instr = self.head_instrumentation(z)
        logits_rhythm = self.head_rhythmic_density(z)
        
        val_tension = torch.sigmoid(self.head_tension_level(z))
        val_swell = torch.sigmoid(self.head_dynamic_swell(z))
        
        return {
            "kl_loss": kl_divergence,
            "categorical": {
                "tempo": logits_tempo.view(bs, num_scenes, -1),
                "harmony": logits_harmony.view(bs, num_scenes, -1),
                "texture": logits_texture.view(bs, num_scenes, -1),
                "instrumentation": logits_instr.view(bs, num_scenes, -1),
                "rhythmic_density": logits_rhythm.view(bs, num_scenes, -1)
            },
            "continuous": {
                "tension": val_tension.view(bs, num_scenes, 1),
                "dynamic_swell": val_swell.view(bs, num_scenes, 1)
            }
        }
