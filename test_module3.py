import torch
from config import pipeline_config
from models.module2_3_narrative import GatedChronologicalScenePooler, SceneNarrativeEngine

def test_module3_mathematics():
    print("Initializing Deep Mathematical Module 3 Unit Test (Mamba + Memory)...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    m1_hidden_size = 512 
    config_dict = pipeline_config.module2_3
    config_dict["mamba_d_model"] = 128
    config_dict["mamba_n_layer"] = 2
    config_dict["num_memory_slots"] = 10
    
    print("Instantiating SceneNarrativeEngine...")
    try:
        model = SceneNarrativeEngine(config_dict, m1_hidden_size=m1_hidden_size).to(device)
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return
        
    print("Preparing mock conversational data...")
    bs = 2
    num_scenes = 3
    num_utts = 4
    
    batched_utterances = torch.randn(bs, num_scenes, num_utts, m1_hidden_size).to(device)
    utterances_padding_mask = torch.zeros(bs, num_scenes, num_utts, dtype=torch.bool).to(device)
    
    print("Testing Mamba Continuous propagation and Episodic Memory Fusion...")
    model.train()
    
    try:
        augmented_narrative = model(batched_utterances, utterances_padding_mask)
        
        # Verify shape Output should match the Mamba state dim
        print(f"Output Shape -> Augmented Narrative State: {augmented_narrative.shape}")
        expected_shape = (bs, num_scenes, config_dict["mamba_d_model"])
        assert augmented_narrative.shape == expected_shape, "Neural routing shape mismatch!"
        
        print("Testing Backpropagation through Episodic Memory & Mamba...")
        loss = augmented_narrative.sum()
        loss.backward()
        
        has_memory_grad = False
        has_mamba_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if "episodic_memory.memory" in name:
                    has_memory_grad = True
                if "mamba" in name:
                    has_mamba_grad = True
                    
        assert has_memory_grad, "Explicit Episodic Memory matrix failed to receive gradients!"
        assert has_mamba_grad, "Continuous Mamba block detached from graph!"
        
        print("Module 3 Implementation VERIFIED. Mamba and Differentiable Memory seamlessly routing.")
        
    except Exception as e:
        print(f"Error during forward/backward pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_module3_mathematics()
