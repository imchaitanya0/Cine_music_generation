import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import json
import os

class CineEmotionDataset(Dataset):
    """
    Advanced Dataset handler for Screenplay-to-Music data.
    Parses a DIRECTORY of Movie JSONs, extracting the `annotations` (scenes)
    and formatting them into the 4D Tensor required by the Mamba/Scene Pooler architecture:
    [num_scenes, num_utterances, seq_length]
    """
    def __init__(self, data_dir, limit=None, max_scenes=20, max_utts=15, max_len=128, tokenizer_name="microsoft/deberta-v3-base"):
        self.max_scenes = max_scenes
        self.max_utts = max_utts
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add Special Structural Tokens
        special_tokens_dict = {'additional_special_tokens': ['[SCENE]', '[SPK]', '[TXT]', '[ACT]']}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        
        self.data_dir = data_dir
        
        # Grab all JSONs and limit them to save Kaggle training time
        if not os.path.exists(self.data_dir):
            print(f"⚠️ Warning: Dataset dir not found at {self.data_dir}.")
            self.files = []
        else:
            self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.json')]
            if limit:
                self.files = self.files[:limit]
                print(f"Restricting dataset to {limit} files for testing.")
        
    def __len__(self):
        return len(self.files) if len(self.files) > 0 else 1

    def __getitem__(self, idx):
        # Return Dummy Tensors if checking math without files
        if len(self.files) == 0:
            return {
                "input_ids": torch.randint(0, 1000, (self.max_scenes, self.max_utts, self.max_len)),
                "attention_mask": torch.ones((self.max_scenes, self.max_utts, self.max_len)),
                "tension_level": torch.rand(self.max_scenes, 1),
                "harmony": torch.randint(0, 7, (self.max_scenes,))
            }
            
        file_path = self.files[idx]
        with open(file_path, 'r') as f:
            movie_data = json.load(f)
            
        annotations = movie_data.get("annotations", [])
        
        # Truncate to max_scenes
        annotations = annotations[:self.max_scenes]
        
        # Initialize Tensors
        input_ids = torch.zeros((self.max_scenes, self.max_utts, self.max_len), dtype=torch.long)
        attention_mask = torch.zeros((self.max_scenes, self.max_utts, self.max_len), dtype=torch.float)
        tension_levels = torch.zeros((self.max_scenes, 1), dtype=torch.float)
        harmony_labels = torch.zeros((self.max_scenes,), dtype=torch.long)
        
        for s_idx, scene in enumerate(annotations):
            # Map Continuous Label
            tension_levels[s_idx, 0] = float(scene.get("tension_level", 0.0)) / 10.0 # Normalize 0-1
            
            # Simulated Map of Categorical Harmony
            h_style = scene.get("harmonic_style", "diatonic")
            harmony_labels[s_idx] = 1 if h_style == "chromatic" else 0
            
            # Simple text split to simulate utterances
            raw_text = scene.get("scene_text", "")
            utterances = [u for u in raw_text.split('\n\n') if len(u.strip()) > 0]
            utterances = utterances[:self.max_utts]
            
            for u_idx, utt in enumerate(utterances):
                encoded = self.tokenizer(
                    utt,
                    padding="max_length",
                    truncation=True,
                    max_length=self.max_len,
                    return_tensors="pt"
                )
                input_ids[s_idx, u_idx] = encoded["input_ids"].squeeze(0)
                attention_mask[s_idx, u_idx] = encoded["attention_mask"].squeeze(0)
                
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "tension_level": tension_levels,
            "harmony": harmony_labels
        }

def process_and_get_loaders(data_dir, batch_size=4, limit=None):
    dataset = CineEmotionDataset(data_dir=data_dir, limit=limit)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    return loader, dataset.tokenizer
