from utils import *

if __name__ == "__main__":
    # define sizes for training, validation, and testing
    N_train = 10_000
    N_val = 1_000
    N_test = 2_000
    
    # load training subset
    balanced_train_subset, stop_idx_train = get_balanced_imagenet_subsets("timm/imagenet-1k-wds", N_train, num_classes=1000, split="train", start_idx=0)
    
    # load validation subset
    balanced_val_subset, _ = get_balanced_imagenet_subsets("timm/imagenet-1k-wds", N_val, num_classes=1000, split="train", start_idx=stop_idx_train+1)
    
    # load test subset
    balanced_test_subset, _ = get_balanced_imagenet_subsets("timm/imagenet-1k-wds", N_test, num_classes=1000, split="validation", start_idx=0)
    
    # save datasets
    save_dataset(balanced_train_subset, "training", N_train)
    save_dataset(balanced_val_subset, "validation", N_val)
    save_dataset(balanced_test_subset, "testing", N_test)