from torchvision import datasets
import torch
from utils import *
from datasets import load_dataset, Dataset, Features, Value, Image, concatenate_datasets
from torch.utils.data import Subset
from transformers import AutoImageProcessor, AutoModelForImageClassification



def get_checkpoint_model(alpha, beta, sae_expansion_factor):
    '''
    Returns loaded vit handler and sae for given params
    '''
    sae_input_dim = 768 # change if needed
    latent_dim = sae_expansion_factor * sae_input_dim
    
    vit_handler = ViTHandler('google/vit-base-patch16-224', 'google/vit-base-patch16-224')
    sae = SAE(sae_input_dim, latent_dim)
    
    checkpoint_path = f"./checkpoints/alpha_{alpha}_beta_{beta}_sae_expansion_factor_{sae_expansion_factor}"
    sae_state_dict = torch.load(checkpoint_path + "/sae.pt", map_location=get_device())
    vit_state_dict = torch.load(checkpoint_path + "/vit.pt", map_location=get_device())
    
    vit_handler.base_model.load_state_dict(vit_state_dict)
    sae.load_state_dict(sae_state_dict)
    
    vit_handler.base_model.to(get_device())
    sae.to(get_device())
    return vit_handler, sae

def evaluate_model(model, test_ds, batch_size=64):
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    
    model.eval()
    
    correct = 0    
    with torch.no_grad():        
        for batch in test_loader:
            X = batch['img'].to(get_device()) # X: [B, channels, h, w]
            Y = batch['class'].to(get_device()) # Y: [B]
            
            logits = model(X).logits # logits: [B, num_classes]
            predictions = torch.argmax(logits, dim=-1)
            print(predictions.shape, Y.shape)
            correct += torch.sum(predictions==Y).item()
    
    return correct/len(test_ds)
            
if __name__=="__main__":
    # perform a sweep of alpha, beta, and sae_expansion_factor
    alpha_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    beta_values = [1.0]
    sae_expansion_factor_values = [2, 4]
    
    # DATASETS
    temp_processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224')
    
    # Imagenetv2
    imagenetv2 = datasets.ImageFolder(root='../imagenetv2-matched-frequency-format-val/')
    imagenetv2_dict = {"jpg": [img for (img, label) in imagenetv2], "cls": [label for (img, label) in imagenetv2]}
    imagenetv2_ds = Dataset.from_dict(imagenetv2_dict)
    imagenetv2_ds = imagenetv2_ds.set_transform(functools.partial(preprocess_data, processor=temp_processor), remove_columns=["jpg", "cls"])
    imagenetv2_ds.set_format(type="torch", columns=["img", "class"])

    # Imagenet
    test_dataset_path = "DeepLearningProj/imagenet_testing_subset_2000_balanced"
    test_ds = load_dataset(test_dataset_path, split="train")
    test_ds = test_ds.map(functools.partial(preprocess_data, processor=temp_processor), remove_columns=["jpg", "cls"])
    test_ds.set_format(type="torch", columns=["img", "class"])
    
    test_ds = Subset(test_ds, range(3))# TODO: change
    hyperparam_accuracies = []
    
    lr = 0.001 # TODO: change if needed
    
    for alpha in alpha_values:
        for beta in beta_values:
            for sae_expansion_factor in sae_expansion_factor_values:
                vit_handler, sae = get_checkpoint_model(alpha, beta, sae_expansion_factor)
                accuracy = evaluate_model(vit_handler.base_model, test_ds)
                robustness = evaluate_model(vit_handler.base_model, imagenetv2_ds)
                print("DONE")
                hyperparam_accuracies.append({"alpha": alpha, "beta": beta, "lr": lr, "expansion": sae_expansion_factor, "accuracy": accuracy})
    
    print(hyperparam_accuracies)
                
    