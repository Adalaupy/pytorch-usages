from models.cnn_model import CNN_Model
from utils import (
    EarlyStopping,
    EpochTrainer,
    Face_Detector,
    ResizeKeepRatioPad,
    get_device,
    get_mean_std,
    create_preprocess_config,
    build_cnn_transform,
    build_augmentation_transform,
    get_image_dataset,
    kaggle_download,

)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms


EVAL_METHOD = 'Accuracy'
TEST_SIZE = 0.2
KERNEL_SIZE = 3
PADDING = 1
POOL_SIZE = 2

# ================================================================================================
# Get data, Detect Face(if needed) + Resize image + Get number of labels
# ================================================================================================

def init_data_transform(kaggle_path, input_size, isFace):
    
    # Prepare Initial image transform
    stats_steps = []
    if isFace:
        stats_steps.append(Face_Detector(input_size=input_size))

    stats_steps.extend([
        ResizeKeepRatioPad(input_size, fill=0),
        transforms.ToTensor(),
    ])

    stats_transform = transforms.Compose(stats_steps)


     # Get Data from kaggle
    local_path = kaggle_download(kaggle_path, enter_first_folder=True)

   
    full_dataset = get_image_dataset(local_path, stats_transform)

    class_names = full_dataset.classes
    num_output = len(class_names)



    return full_dataset, class_names, num_output


# ================================================================================================
# Prepare augmentation for training data and testing data respectively
# ================================================================================================

def augmentation( train_dataset, test_dataset, input_size, isFace, batch_size):
    
    # Statistic of training data
    mean, std    = get_mean_std(train_dataset)
    sample_x, _  = train_dataset[0]
    channel_size = sample_x.shape[0]


    # Augmentation for Training data
    train_augmentation = build_augmentation_transform(
        horizontal_flip    = True,
        rotation_degrees   = 15,
        color_jitter       = True,
        random_grayscale_p = 0.1,
    )   
    
    
    # Augmentation for both
    preprocess_config = create_preprocess_config(
        input_size=input_size,
        mean=mean,
        std=std,
        channel_size=channel_size,
        keep_ratio_pad=True,
        pad_fill=0,
    )
    norm_transform = build_cnn_transform(preprocess_config)


    train_transform = norm_transform.transforms + train_augmentation.transforms
    test_transform  = norm_transform.transforms


    # Add face detect as transform if IsFace = True
    if isFace:
        
        train_transform = [Face_Detector(input_size=input_size)] + train_transform
        test_transform  = [Face_Detector(input_size=input_size)] + test_transform


    train_dataset.dataset.transform = transforms.Compose( train_transform )
    train_dataset.dataset.transform = transforms.Compose( test_transform  )


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    return channel_size, preprocess_config, train_loader, test_loader


# ================================================================================================
# Handle dataset split 
# ================================================================================================

def split_data(full_dataset):
    

    test_cnt  = int(TEST_SIZE * len(full_dataset) )
    train_cnt = len(full_dataset) - test_cnt

    train_dataset, test_dataset  = random_split(    
        full_dataset, 
        [train_cnt,test_cnt],
        generator=torch.Generator().manual_seed(42)
    )


    return train_dataset, test_dataset


# ================================================================================================
# main
# ================================================================================================

def main_face_recognition(
     batch_size   = 40
    ,epochs       = 20
    ,hidden_size  = 16
    ,stride       = 1
    ,input_size   = (100, 100)
    ,patience     = 5
    ,lr           = 0.001
    ,isFace       = True
    ,kaggle_data_path = "vishesh1412/celebrity-face-image-dataset"
    ,output_path = '../checkpoints/face_recognition_checkpoint.pt'
):

    # Define device
    device = get_device()
    

    # Get data
    full_dataset, class_names, num_output = init_data_transform(kaggle_data_path, input_size, isFace)
    
    # Split data into training, testing
    train_dataset, test_dataset = split_data(full_dataset)


    # Apply Augmentation to the dataset
    channel_size, preprocess_config, train_loader, test_loader = augmentation( train_dataset, test_dataset, input_size, isFace, batch_size)


    # Prepare config of the model
    model_config = {
        "channel_size": channel_size,
        "input_size": tuple(input_size),
        "hidden_size": hidden_size,
        "num_output": num_output,
        "kernel_size": KERNEL_SIZE,
        "stride": stride,
        "padding": PADDING,
        "pool_size": POOL_SIZE,
    }



    # Prepare model stuff
    model = CNN_Model(**model_config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    early_stopping = EarlyStopping(
        patience=patience,
        path=f"../checkpoints/{output_path}.pt",
        checkpoint_data={
            "model_config": model_config,
            "preprocess_config": preprocess_config,
            "class_names": class_names,
        },
    )


    # Start training
    epoch_trainer = EpochTrainer(
        model = model,
        early_stopping = early_stopping,
        device = device,
        optimizer = optimizer,
        criterion = criterion,
        eval_method = EVAL_METHOD
    )
    for epoch in range(epochs):

        avg_train_loss, avg_val_loss, result = epoch_trainer(train_loader , test_loader )

        for key, value in result.items():
            
            print(f"Epoch {epoch + 1:3d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | {key}: {value:.4f}")
        
        # ==================== Early Stopping Check ====================

        early_stopping(avg_val_loss, model)

        if early_stopping.early_stop:
            
            print("Early stopping triggered! Training stopped.")
            
            break