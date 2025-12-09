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

def get_bin_index(num_bins, value):
    if value == 1.0:
        return num_bins-1
    for i in range(num_bins):
        bin_start = i/num_bins
        bin_end = (i+1)/num_bins
        if bin_start <= value < bin_end:
            return i 
        
    raise Exception("value not in [0, 1]")
    
def evaluate_model(model, test_ds, ECE_bins=15, batch_size=64):
    '''
    calculates accuracy and ECE using ECE_bins number of evenly split bins
    '''
    
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    device = get_device()
    
    model.eval()
    
    total_correct = 0 # for accuracy calculation
    binned_acc = [0]*ECE_bins # number correct per bin for ECE calculation
    binned_conf = [0]*ECE_bins # sum of max probability per bin for ECE calculation
    bin_size = [0]*ECE_bins # elements in bin
    
    with torch.no_grad():
        for batch in test_loader:
            X = batch['img'].to(device) # X: [B, channels, h, w]
            Y = batch['class'].to(device) # Y: [B]
            
            logits = model(X).logits # logits: [B, num_classes]
            predictions = torch.argmax(logits, dim=-1)
            max_probs = torch.max(torch.softmax(logits, dim=-1), dim=-1).values # max probabilites for each batch: [B]
            
            correctness = (predictions==Y)
            
            for i in range(max_probs.shape[0]):
                bin_idx = get_bin_index(ECE_bins, max_probs[i])
                binned_acc[bin_idx] += correctness[i].item()
                binned_conf[bin_idx] += max_probs[i].item()
                bin_size[bin_idx] += 1
            
            total_correct += torch.sum(correctness).item()
    
    # calculate overall accuracy
    accuracy = total_correct/len(test_ds)
    
    # calculate ECE
    ECE = 0
    for bin_num in range(ECE_bins):
        sz = bin_size[bin_num]
        if sz == 0:
            continue
        ECE += (sz/len(test_ds))*abs(binned_acc[bin_num]/sz - binned_conf[bin_num]/sz)
    
    return accuracy, ECE
            
if __name__=="__main__":
    # perform a sweep of alpha, beta, and sae_expansion_factor
    alpha_values = [-1.0, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    beta_values = [1.0]
    sae_expansion_factor_values = [2, 4, 32]
    
    # processor for datasets
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
    
    # ImageNet
    test_dataset_path = "DeepLearningProj/imagenet_testing_subset_2000_balanced"
    non_blurred_imagenet_test_ds = load_dataset(test_dataset_path, split="train")
    imagenet_test_ds = non_blurred_imagenet_test_ds.map(functools.partial(preprocess_data, processor=temp_processor), remove_columns=["jpg", "cls"])
    imagenet_test_ds.set_format(type="torch", columns=["img", "class"])
    
    # Blurred ImageNet
    blurred_imagenet_test_ds = non_blurred_imagenet_test_ds.map(functools.partial(preprocess_data, processor=temp_processor, blur=True), remove_columns=["jpg", "cls"])
    blurred_imagenet_test_ds.set_format(type="torch", columns=["img", "class"])
    
    #save example of blurred vs. non-blurred
    non_blurred_0 = non_blurred_imagenet_test_ds[0]["jpg"].convert("RGB")
    non_blurred_1 = non_blurred_imagenet_test_ds[1]["jpg"].convert("RGB")
    blurred_0 = non_blurred_imagenet_test_ds[0]["jpg"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    blurred_1 = non_blurred_imagenet_test_ds[0]["jpg"].convert("RGB").filter(ImageFilter.GaussianBlur(radius=3))
    
    non_blurred_0.save("non_blurred_0.jpg")
    non_blurred_1.save("non_blurred_1.jpg")
    blurred_0.save("blurred_0.jpg")
    blurred_1.save("blurred_1.jpg")
    
    print("saved blurred and non-blurred examples")
        
    
    # Evaluate all checkpointed models 
    hyperparam_results = []
    lr = 0.001 # TODO: change if needed
    for alpha in alpha_values:
        for beta in beta_values:
            for sae_expansion_factor in sae_expansion_factor_values:
                vit_handler, sae = get_checkpoint_model(alpha, beta, sae_expansion_factor)
                imagenet_1k_accuracy, imagenet_1k_ECE = evaluate_model(vit_handler.base_model, imagenet_test_ds)
                imagenet_v2_accuracy, imagenet_v2_ECE = evaluate_model(vit_handler.base_model, imagenetv2_ds)
                blurred_imagenet_1k_accuracy, blurred_imagenet_1k_ECE = evaluate_model(vit_handler.base_model, blurred_imagenet_test_ds)
                print("DONE")
                hyperparam_results.append({"alpha": alpha, "beta": beta, "lr": lr, "expansion": sae_expansion_factor, "imagenet_1k accuracy": imagenet_1k_accuracy, "imagenet_1k ECE": imagenet_1k_ECE, "imagenet_v2 accuracy": imagenet_v2_accuracy, "imagenet_v2 ECE": imagenet_v2_ECE, "blurred_imagenet_1k accuracy": blurred_imagenet_1k_accuracy, "blurred_imagenet_1k ECE": blurred_imagenet_1k_ECE})
    
    df = pd.DataFrame(hyperparam_results)
    df.to_csv('FULL_RESULTS.csv', index=False)
                
    