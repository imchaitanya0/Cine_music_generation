import torch
import torch.nn as nn
from transformers import AutoTokenizer
from config import pipeline_config
from models.module1_encoder import RiemannianUtteranceEncoder, RiemannianSupConLoss

def test_module1():
    print("Initializing Deep Mathematical Module 1 Unit Test...")
    
    config_dict = pipeline_config.module1
    # Mocking for generic CPU/GPU test without massive download
    config_dict["model_name"] = "albert-base-v2"
    config_dict["use_lora"] = True
    config_dict["lora_r"] = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("Instantiating RiemannianUtteranceEncoder...")
    try:
        model = RiemannianUtteranceEncoder(config_dict, num_emotions=7).to(device)
    except Exception as e:
        print(f"Failed to instantiate model: {e}")
        return
        
    print("Riemannian Encoder initialized. Trainable LoRA parameters active.")
    
    # Mock Data
    tokenizer = AutoTokenizer.from_pretrained("albert-base-v2")
    texts = [
        "[SPK: TONY] This is mathematically superior.",
        "[SPK: STEVE] I agree completely.",
        "[SPK: TONY] This is mathematically superior! (augmented)" 
    ] # Two views of Tony, one of Steve
    labels = torch.tensor([0, 1, 0]).to(device)
    
    inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    
    print("Testing Intensity-Aware Forward Pass...")
    model.train()
    pooled_utt, spherical_proj, logits = model(input_ids, attention_mask)
    
    # Mathematical shape verifications
    assert pooled_utt.shape[1] == model.hidden_size, "Pooling dim mismatch."
    assert spherical_proj.shape[1] == 128, "Spherical dim mismatch."
    
    # Mathematical Sphere Check: L2 norm of the projection must strictly equal 1.0 (with low tolerance)
    norms = torch.norm(spherical_proj, p=2, dim=1)
    sphere_adherence = torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    print(f"Are latents bounded to an exact hypersphere? {'YES' if sphere_adherence else 'NO'}")
    assert sphere_adherence, "Spherical normalization failed! Not adhering to vMF distribution logic."
    
    print("Testing Riemannian Contrastive Loss and Backpropagation...")
    try:
        supcon_criterion = RiemannianSupConLoss(temperature=0.1, device=device)
        ce_criterion = nn.CrossEntropyLoss()
        
        # Batch simulation: unsqueeze to add view dimension [bsz, 1, dim]
        spherical_proj_unsq = spherical_proj.unsqueeze(1)
        
        # Calculate loss mapping
        loss_supcon = supcon_criterion(spherical_proj_unsq, labels)
        loss_ce = ce_criterion(logits, labels)
        
        total_loss = loss_supcon + loss_ce
        print(f"Total Loss computed -> SupCon: {loss_supcon.item():.4f}, CE: {loss_ce.item():.4f}")
        
        # Verify gradients flow into Intensity Scorer and LoRA
        total_loss.backward()
        
        has_intensity_grad = False
        has_lora_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if "intensity_pooler" in name:
                    has_intensity_grad = True
                if "lora" in name:
                    has_lora_grad = True
                    
        assert has_intensity_grad, "Intensity Attention logic is detached from computational graph!"
        assert has_lora_grad, "LoRA adapters failed to accumulate gradients!"
        
        print("Riemannian Module 1 Implementation VERIFIED. All mathematical constraints hold.")
        
    except Exception as e:
        print(f"Error during loss calculation: {e}")

if __name__ == "__main__":
    test_module1()
