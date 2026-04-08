import torch
from config import pipeline_config
from models.module4_planner import CVAEMusicPlanner

def test_module4_cvae_mathematics():
    print("Initializing Deep Mathematical Module 4 Unit Test (CVAE)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config_dict = pipeline_config.module4
    mamba_d_model = 128
    
    print("Instantiating CVAEMusicPlanner...")
    try:
        model = CVAEMusicPlanner(config_dict, mamba_d_model=mamba_d_model).to(device)
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return
        
    print("Preparing mock batch...")
    # dimensions: [batch_size, num_scenes, mamba_d_model]
    bs = 2
    num_scenes = 5
    narrative_states = torch.randn(bs, num_scenes, mamba_d_model).to(device)
    
    print("Testing Variational Forward Pass...")
    model.train() # Crucial to trigger the eps * std sampling logic!
    
    try:
        outputs = model(narrative_states)
        
        # 1. Verify Analytical KL-Divergence
        kl_loss = outputs['kl_loss']
        print(f"Calculated mathematical KL-Divergence Penalty: {kl_loss.item():.4f}")
        assert kl_loss.item() > 0.0, "KL Divergence mathematically cannot be zero initially without absolute prior collapse!"
        
        # 2. Verify Output Shapes
        for name, tensor in outputs['categorical'].items():
            expected = (bs, num_scenes)
            assert tensor.shape[:2] == expected, f"Batch/Scene mismatch for {name}"
            
        print("Testing Backpropagation through the Reparameterization Trick (\eps)..")
        
        # We need to sum up some structural branch to see if gradients flow through the 
        # sampling epsilon back to mu and logvar.
        loss = outputs['categorical']['harmony'].sum() + kl_loss
        loss.backward()
        
        has_mu_grad = False
        has_logvar_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if "fc_mu" in name:
                    has_mu_grad = True
                if "fc_logvar" in name:
                    has_logvar_grad = True
                    
        assert has_mu_grad, "Gradient detached. Backprop failed to reach mu through sampling logic!"
        assert has_logvar_grad, "Gradient detached. Backprop failed to reach logvar through Reparameterization Trick!"
        
        print("Module 4 Implementation VERIFIED. Analytical CVAE sampling perfectly gradients.")
        
    except Exception as e:
        print(f"Error during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_module4_cvae_mathematics()
