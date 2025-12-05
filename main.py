from utils import *
from datasets import load_dataset, Dataset
import wandb

if __name__ == "__main__":
    # fine_tuning_data = get_imagenet_subset("misc", 100) 
    # fine_tuning_data = get_imagenet_subset("finetune", 5000)
    # val_data = get_imagenet_subset("validation", 500, 5000, 5_500)
    
    print(f"DEVICE = {get_device()}")

    vit_handler = ViTHandler('google/vit-base-patch16-224', 'google/vit-base-patch16-224')
    target_module = "vit.encoder.layer.11.output" # dim 3072

    print(vit_handler.get_module(target_module))
    sae_input_dim = vit_handler.get_module(target_module).dense.out_features
    sae_expansion_factor=2
    latent_dim = sae_expansion_factor*sae_input_dim

    layer_11_sae = SAE(sae_input_dim, latent_dim)
    vit_handler.register_hook(target_module)
    vit_handler.freeze_all_layers_up_to(10)

    train_dataset = "DeepLearningProj/imagenet_finetune_subset_5000"
    fine_tuning_data = load_data_hf(train_dataset, split="train")
    fine_tuning_data = fine_tuning_data.map(functools.partial(preprocess_data, processor=vit_handler.processor), remove_columns=["jpg", "cls"])
    fine_tuning_data.set_format(type="torch", columns=["img", "class"])
    
    val_dataset = "DeepLearningProj/imagenet_validation_subset_500"
    val_tuning_data = load_data_hf(val_dataset, split="train")
    val_tuning_data = val_tuning_data.map(functools.partial(preprocess_data, processor=vit_handler.processor), remove_columns=["jpg", "cls"])
    val_tuning_data.set_format(type="torch", columns=["img", "class"])
    
    num_epochs = 8
    lr=0.01
    batch_size = 128
    alpha = 0.7
    beta = 1
    vit_loss_fn = nn.CrossEntropyLoss()
    sae_loss_fn = SAELoss(beta=beta)
    no_freeze=10

    # create new wandb run
    run = wandb.init(
        entity="hann-yun-gao",
        project="6.7960_proj",
        config={
            "finetune_dataset": train_dataset,
            "validation_dataset": val_dataset,
            "lr": lr,
            "sae_layer": target_module,
            "sae_input_dim": sae_input_dim,
            "sae_expansion_factor": sae_expansion_factor,
            "optimizer": "Adam",
            "epochs": num_epochs,
            "batch_size": batch_size,
            "no_freeze": 10,
            "alpha": alpha,
            "beta": beta,
        }
    )
    
    fine_tune(run, vit_handler, layer_11_sae, fine_tuning_data, val_tuning_data, num_epochs, lr, batch_size, vit_loss_fn, sae_loss_fn, alpha, no_freeze)
    vit_handler.clear_hooks()