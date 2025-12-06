from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import functools
from datasets import load_dataset, Dataset, Features, Value, Image, concatenate_datasets
from itertools import islice
from tqdm import tqdm


def preprocess_data(sample, processor=None):
    image_as_tensor = processor(images=sample['jpg'].convert("RGB"), return_tensors="pt")
    return {'img': image_as_tensor['pixel_values'].squeeze(0), 'class': int(sample['cls'])}
    
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_balanced_imagenet_subsets(dataset, N, num_classes, split, start_idx):
    '''
    dataset: dataset name
    start_idx: where to start streaming from
    N: subset size
    split: train or validation
    
    Returns balanced subset and the index the dataset stopped at
    '''
    ds = load_dataset(dataset, split=split, streaming=True)
    assert N%num_classes == 0, "Dataset size not divisible by number of classes"
    num_samples_per_class = N//num_classes
    
    num_samples_needed_per_class = {i: num_samples_per_class for i in range(num_classes)}
    samples_list = []
    for idx, sample in enumerate(islice(ds, start_idx, None)):
        sample_class = sample['cls']
        if num_samples_needed_per_class[sample_class] > 0:
            num_samples_needed_per_class[sample_class] -= 1
            samples_list.append(sample)
            
            # check if we're done
            if all(num_samples_needed_per_class[label] == 0 for label in num_samples_needed_per_class.keys()):
                break
    
    return Dataset.from_list(samples_list), start_idx + idx

def save_dataset(ds, purpose, N):
    '''
    ds: dataset
    purpose: purpose of dataset
    N: dataset size
    '''
    hf_path = f"DeepLearningProj/imagenet_{purpose}_subset_{N}_balanced"
    ds.push_to_hub(hf_path, private=True)
    print(f"Saved to {hf_path}")

class SAE(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.encoder = nn.Linear(dim, latent_dim)
        self.act = nn.ReLU()
        self.decoder = nn.Linear(latent_dim, dim)
    def forward(self, x):
        x_enc = self.encoder(x)
        z = self.act(x_enc)
        x_dec = self.decoder(z)
        return z, x_dec

class SAELoss(nn.Module):
    def __init__(self, beta):
        super().__init__()
        assert 0 <= beta <= 1, f"Beta={beta} out of bounds"
        self.beta = beta
        self.mse_loss = nn.MSELoss()
    
    def forward(self, z, latent_activations, z_prime):
        recon_loss = self.mse_loss(z, z_prime)
        latent_dim = latent_activations.shape[-1]
        sparsity = (latent_activations.norm(p=1, dim=-1)/latent_dim).mean()
        return recon_loss + self.beta * sparsity

def calculate_loss(X, Y, vit_handler, vit_loss_fn, sae_model, sae_loss_fn):
    vit_logits = vit_handler.base_model(X).logits
    print(f"{vit_logits.shape=}") # logits: [B, num_classes]
    CE_loss = vit_loss_fn(vit_logits, Y)

    if vit_handler.hook is not None:
        z = vit_handler.hook
        print(f"{z.shape=}") # z: [B, num_patches, model_dim]
        # print(f"{sae_module=}") # maps model_dim to model_dim*2

        #normalization like in Su et. al
        z_centered = (z - torch.mean(z, dim=1, keepdim=True))
        z_normalized = z_centered / torch.norm(z_centered, dim=2, keepdim=True)
        print(f"{z_normalized.shape=}")
        latent_activations, z_recon = sae_model(z_normalized)

        print(f"{latent_activations.shape=}, {z_recon.shape=}") # latent_activations: [B, num_patches, model_dim*2]     z_recon: [B, num_patches, model_dim]
        
        SAE_loss = sae_loss_fn(z, latent_activations, z_recon)
    else:
        print("NO HOOKS FOUND")
        SAE_loss = 0
    
    return CE_loss, SAE_loss

def fine_tune(wandb_run, vit_handler, sae_module, train_dataset, val_dataset, num_epochs, lr, batch_size, vit_loss_fn, sae_loss_fn, alpha, no_freeze):
    wandb_log = {}

    vit_handler.base_model.to(get_device()) 
    sae_module.to(get_device())
    vit_handler.freeze_all_layers_up_to(no_freeze)

    all_params = list(vit_handler.base_model.parameters()) + list(sae_module.parameters())
    non_frozen_params = [p for p in all_params if p.requires_grad]
    optimizer = torch.optim.Adam(non_frozen_params, lr)

    # print(train_dataset[0]['img'].shape, train_dataset[0]['class'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    train_loss = 0
    step_tracker = 0
    for i in tqdm(range(num_epochs)):

        # calculate validation loss
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
        vit_handler.base_model.eval()
        sae_module.eval()
        num_iters = 0
        with torch.no_grad():
            for batch in val_loader:
                assert num_iters < 1
                X = batch['img'].to(get_device())
                Y = batch['class'].to(get_device())

                val_CE_loss, val_SAE_loss = calculate_loss(X, Y, vit_handler, vit_loss_fn, sae_module, sae_loss_fn)
                val_combined_loss = (alpha) * val_CE_loss + (1-alpha) * val_SAE_loss

                wandb_log["val_CE_loss"] = val_CE_loss.item()
                wandb_log["val_SAE_loss"] = val_SAE_loss.item()
                wandb_log["val_loss"] = val_combined_loss.item()

                num_iters += 1 # sanity check that we only have 1 batch for validation

            wandb_run.log(wandb_log, step=step_tracker) 

        # set to train mode
        vit_handler.base_model.train()
        sae_module.train()

        for batch in train_loader:
            X = batch['img'].to(get_device())
            Y = batch['class'].to(get_device())

            print(f"{X.shape=}\n{Y.shape=}") # X: [B, channels, h, w]   Y: [B]

            vit_logits = vit_handler.base_model(X).logits
            print(f"{vit_logits.shape=}") # logits: [B, num_classes]
            
            train_CE_loss, train_SAE_loss = calculate_loss(X, Y, vit_handler, vit_loss_fn, sae_module, sae_loss_fn)
            
            assert 0 <= alpha <= 1, "Alpha out of bounds"
            train_combined_loss = (alpha) * train_CE_loss + (1-alpha) * train_SAE_loss

            wandb_log["train_CE_loss"] = train_CE_loss.item()
            wandb_log["train_SAE_loss"] = train_SAE_loss.item()
            wandb_log["train_loss"] = train_combined_loss.item()

            optimizer.zero_grad()
            train_combined_loss.backward()
            optimizer.step()
            
            print(f"{train_combined_loss=}")

            wandb_run.log(wandb_log, step=step_tracker)
            step_tracker += 1

    wandb_run.finish()


class ViTHandler():
    def __init__(self, processor_path, model_path):
        self.processor = AutoImageProcessor.from_pretrained(processor_path)
        self.base_model = AutoModelForImageClassification.from_pretrained(model_path)
        self.hook = None
        self.hook_handle = None

    def print_modules(self):
        for name, mod in self.base_model.named_modules():
            print(name, "\n\t", mod)

    def get_module(self, name):
        modules = dict(self.base_model.named_modules())
        module = modules[name]
        return module
    
    def register_hook(self, module_name):
        
        def intermediate_hook_fn(module, input, output):
            self.hook = output
        
        module = self.get_module(module_name)
        self.hook_handle = module.register_forward_hook(intermediate_hook_fn)
        
    def clear_hooks(self):
        if self.hook is not None:
            self.hook_handle.remove()
            self.hook = None
            self.hook_handle = None

    def freeze_all_layers_up_to(self, num_frozen):
        layer_count = 0
        for layer in self.base_model.vit.encoder.layer:
            for param in layer.parameters():
                param.requires_grad = False
            
            layer_count += 1
            if layer_count >= num_frozen:
                break
     
