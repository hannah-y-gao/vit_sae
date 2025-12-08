from utils import *
from datasets import load_dataset, Dataset
import wandb
import argparse  
import functools

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, required=True, help="Project name to save wandb run to")
    parser.add_argument("--epochs", type=int, required=True, help="Maximum number epochs to train for")
    parser.add_argument("--lr", type=float, required=True, help="Learning rate")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha: proportion of weight on CE Loss as opposed to sparsity")
    parser.add_argument("--beta", type=float, required=True, help="Beta: sparsity penalty magnitude")
    parser.add_argument("--no_freeze", type=int, required=True, help="Freeze all layers up to and including this layer")
    parser.add_argument("--expansion_factor", type=int, required=True, help="Expansion factor for SAE")
    parser.add_argument("--target_module", type=str, required=True, help="Module to target for SAE")
    args = parser.parse_args()
    
    print(f"DEVICE = {get_device()}")

    vit_handler = ViTHandler('google/vit-base-patch16-224', 'google/vit-base-patch16-224')
    target_module = args.target_module

    print(vit_handler.get_module(target_module))
    sae_input_dim = vit_handler.get_module(target_module).dense.out_features
    sae_expansion_factor=args.expansion_factor
    latent_dim = sae_expansion_factor*sae_input_dim

    sae = SAE(sae_input_dim, latent_dim)
    vit_handler.register_hook(target_module)
    vit_handler.freeze_all_layers_up_to(10)

    train_dataset_path = "DeepLearningProj/imagenet_training_subset_10000_balanced"
    train_ds = load_dataset(train_dataset_path, split="train")
    train_ds = train_ds.map(functools.partial(preprocess_data, processor=vit_handler.processor), remove_columns=["jpg", "cls"])
    train_ds.set_format(type="torch", columns=["img", "class"])
    
    val_dataset_path = "DeepLearningProj/imagenet_validation_subset_1000_balanced"
    val_ds = load_dataset(val_dataset_path, split="train")
    val_ds = val_ds.map(functools.partial(preprocess_data, processor=vit_handler.processor), remove_columns=["jpg", "cls"])
    val_ds.set_format(type="torch", columns=["img", "class"])
    
    test_dataset_path = "DeepLearningProj/imagenet_testing_subset_2000_balanced"
    test_ds = load_dataset(test_dataset_path, split="train")
    test_ds = test_ds.map(functools.partial(preprocess_data, processor=vit_handler.processor), remove_columns=["jpg", "cls"])
    test_ds.set_format(type="torch", columns=["img", "class"])

    project_name=args.project
    num_epochs = args.epochs
    lr = args.lr
    batch_size = args.batch_size
    alpha = args.alpha
    beta = args.beta
    vit_loss_fn = nn.CrossEntropyLoss(reduction='mean')
    sae_loss_fn = SAELoss(beta=beta)
    no_freeze=args.no_freeze

    # create new wandb run
    run_config = {
        "finetune_dataset": train_dataset_path,
        "validation_dataset": val_dataset_path,
        "testing_dataset": test_dataset_path,
        "lr": lr,
        "sae_layer": target_module,
        "sae_input_dim": sae_input_dim,
        "sae_expansion_factor": sae_expansion_factor,
        "optimizer": "AdamW",
        "epochs": num_epochs,
        "bastch_size": batch_size,
        "no_freeze": no_freeze,
        "alpha": alpha,
        "beta": beta,
    }
    
    desired_keys =  ["alpha", "beta", "sae_expansion_factor"]
    run_name = "_".join(f"{k}_{run_config[k]}" for k in desired_keys)

    run = wandb.init(
        entity="hann-yun-gao",
        project=project_name,
        config=run_config,
        name=run_name,
    )
    
    fine_tune(run, run_name, vit_handler, sae, train_ds, val_ds, test_ds, num_epochs, lr, batch_size, vit_loss_fn, sae_loss_fn, alpha, no_freeze)
    vit_handler.clear_hooks()