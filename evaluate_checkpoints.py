from torchvision import datasets
import torch
from utils import *
from datasets import load_dataset, Dataset, Features, Value, Image, concatenate_datasets
from torch.utils.data import Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification
import pandas as pd


def get_checkpoint_model(alpha, beta, sae_expansion_factor):
    '''
    Returns loaded vit handler and sae for given params
    '''
    device = get_device()
    
    sae_input_dim = 768 # change if needed
    latent_dim = sae_expansion_factor * sae_input_dim
    
    vit_handler = ViTHandler('google/vit-base-patch16-224', 'google/vit-base-patch16-224')
    sae = SAE(sae_input_dim, latent_dim)
    
    checkpoint_path = f"./checkpoints/alpha_{alpha}_beta_{beta}_sae_expansion_factor_{sae_expansion_factor}"
    sae_state_dict = torch.load(checkpoint_path + "/sae.pt", map_location=device)
    vit_state_dict = torch.load(checkpoint_path + "/vit.pt", map_location=device)
    
    vit_handler.base_model.load_state_dict(vit_state_dict)
    sae.load_state_dict(sae_state_dict)
    
    vit_handler.base_model.to(device)
    sae.to(device)
    return vit_handler, sae

def evaluate_model(model, test_ds, batch_size=64):
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    device = get_device()
    
    model.eval()
    
    correct = 0    
    with torch.no_grad():        
        for batch in test_loader:
            X = batch['img'].to(device) # X: [B, channels, h, w]
            Y = batch['class'].to(device) # Y: [B]
            
            logits = model(X).logits # logits: [B, num_classes]
            predictions = torch.argmax(logits, dim=-1)
            correct += torch.sum(predictions==Y).item()
    
    return correct/len(test_ds)
            
if __name__=="__main__":
    # perform a sweep of alpha, beta, and sae_expansion_factor
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    beta_values = [1.0]
    sae_expansion_factor_values = [2, 4]
    
    # DATASETS
    temp_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224', use_fast=True)
    
    # Imagenetv2
    imagenetv2_path = '../imagenetv2-matched-frequency-format-val/'
    imagenetv2 = datasets.ImageFolder(root=imagenetv2_path)
    samples = imagenetv2.samples
    print("Data Loaded")
    num_samples_per_class = 2
    labels_count = {label: 0 for label in range(1000)}
    imagenetv2_dict = {"jpg": [], "cls": []}
    for path, label in samples:
        if labels_count[int(label)] == 2:
            continue
        imagenetv2_dict["jpg"].append(path)
        imagenetv2_dict["cls"].append(int(label))
        labels_count[int(label)] += 1
        if(all(c==2 for c in labels_count.values())):
            break
    print("Subsetted Data")
    imagenetv2_ds = Dataset.from_dict(imagenetv2_dict)
    print("Dict to Dataset Complete")
    imagenetv2_ds = imagenetv2_ds.map(functools.partial(preprocess_batched_data, processor=temp_processor), batched=True, batch_size=64, remove_columns=["jpg", "cls"])
    imagenetv2_ds.set_format(type="torch", columns=["img", "class"])
    print("Mapping complete")
    
    # Imagenet
    # test_dataset_path = "DeepLearningProj/imagenet_testing_subset_2000_balanced"
    # test_ds = load_dataset(test_dataset_path, split="train")
    # test_ds = test_ds.map(functools.partial(preprocess_data, processor=temp_processor), remove_columns=["jpg", "cls"])
    # test_ds.set_format(type="torch", columns=["img", "class"])
    
    hyperparam_accuracies = []
    
    lr = 0.001 # TODO: change if needed
        
    for alpha in alpha_values:
        for beta in beta_values:
            for sae_expansion_factor in sae_expansion_factor_values:
                vit_handler, sae = get_checkpoint_model(alpha, beta, sae_expansion_factor)
                # accuracy = evaluate_model(vit_handler.base_model, test_ds)
                robustness = evaluate_model(vit_handler.base_model, imagenetv2_ds)
                print("DONE")
                hyperparam_accuracies.append({"alpha": alpha, "beta": beta, "lr": lr, "expansion": sae_expansion_factor, "robustness": robustness})
    
    df = pd.DataFrame(hyperparam_accuracies)
    df.to_csv('imagenetv2 robustness.csv', index=False)
                
    