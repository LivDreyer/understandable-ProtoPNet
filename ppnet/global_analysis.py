import os
import re
from argparse import Namespace
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from .helpers import makedir
from .find_nearest import find_k_nearest_patches_to_prototypes
from .log import create_logger
from .preprocess import preprocess_input_function


def save_prototype_original_img_with_bbox(load_img_dir, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end,
                                          color=(0, 255, 255), markers=None):
    # Read the stored original prototype image for a given epoch/index (OpenCV reads BGR)
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), str(index) + '_prototype-img-original.png'))
    # Draw the bounding box around the prototype patch
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1), color, thickness=2)
    # Optionally draw small red dots for any extra markers (e.g., keypoints)
    if markers is not None:
        for marker in markers:
            cv2.circle(p_img_bgr, (int(marker[0]), int(marker[1])), 5, (0, 0, 255), -1)
    # Convert BGR → RGB and scale to [0,1] for matplotlib
    p_img_rgb = p_img_bgr[..., ::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255

    # Save without axes
    plt.axis('off')
    plt.imsave(fname, p_img_rgb)


def run_analysis(args: Namespace):
    # Limit CUDA to the selected GPU(s)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    train_dir = os.path.join(args.dataset, 'train')
    test_dir = os.path.join(args.dataset, 'test')

    model_path = os.path.abspath(args.model)  # ./saved_models/vgg19/003/checkpoints/10_18push0.7822.pth
    # Extract base arch, run name, checkpoints dir, and filename from the path
    model_base_architecture, experiment_run, _, model_name = re.split(r'\\|/', model_path)[-4:]
    # Parse the first integer in the checkpoint name as the epoch number
    start_epoch_number = int(re.search(r'\d+', model_name).group(0))
    # Handle pruned models which have a slightly different folder layout
    if model_base_architecture == 'pruned_prototypes':
        model_base_architecture, experiment_run = re.split(r'\\|/', model_path)[-6:-4]
        model_name = f'pruned_{model_name}'

    # Prepare output folder and logger
    save_analysis_path = os.path.join(args.out, model_base_architecture, experiment_run, model_name, 'global')
    makedir(save_analysis_path)
    log, logclose = create_logger(log_filename=os.path.join(save_analysis_path, 'global_analysis.log'))
    
    log(f'\nLoad model from: {args.model}')
    log(f'Model epoch: {start_epoch_number}')
    log(f'Model base architecture: {model_base_architecture}')
    log(f'Experiment run: {experiment_run}\n')

    # Allowlist classes used inside the checkpoint for safe torch.load
    from torch.serialization import add_safe_globals
    from ppnet.model import PPNet
    add_safe_globals([PPNet])
    try:
        from ppnet.resnet_features import ResNet_features
        add_safe_globals([ResNet_features])
    except Exception:
        try:
            from ppnet.vgg_features import VGG_features
            add_safe_globals([VGG_features])
        except Exception:
            pass

    # Load checkpoint to CPU or GPU as available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppnet = torch.load(args.model, weights_only=False, map_location=device)
    if torch.cuda.is_available():
        ppnet = ppnet.cuda()
        ppnet_multi = torch.nn.DataParallel(ppnet)  # enable multi-GPU forward if available
    else:
        ppnet_multi = ppnet

    # Image size expected by the model and dataloader batch size
    img_size = ppnet_multi.module.img_size
    batch_size = 100

    # Train set (no normalization here; preprocessing is applied later when needed)
    train_dataset = datasets.ImageFolder(
        train_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)

    # Test set (same as train: keep unnormalized tensors)
    test_dataset = datasets.ImageFolder(
        test_dir,
        transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
        ]))
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False)

    # Where to store nearest-patch visualizations for train/test
    root_dir_for_saving_train_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'train')
    root_dir_for_saving_test_images = os.path.join(save_analysis_path, 'nearest_prototypes', 'test')
    makedir(root_dir_for_saving_train_images)
    makedir(root_dir_for_saving_test_images)

    # Load pre-saved prototype bounding boxes (generated during training/push)
    load_img_dir = os.path.join(os.path.dirname(args.model), '..', 'img')
    assert os.path.exists(load_img_dir), f'Folder "{load_img_dir}" does not exist'
    prototype_info = np.load(os.path.join(load_img_dir, f'epoch-{start_epoch_number}', 'bb.npy'))

    # Save each prototype’s original image with its bbox overlay (train/test copies)
    for j in tqdm(range(ppnet.num_prototypes), desc='Saving learned prototypes'):
        makedir(os.path.join(root_dir_for_saving_train_images, str(j)))
        makedir(os.path.join(root_dir_for_saving_test_images, str(j)))
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_train_images, str(j), 'prototype_bbox.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255)
        )
        save_prototype_original_img_with_bbox(
            load_img_dir=load_img_dir,
            fname=os.path.join(root_dir_for_saving_test_images, str(j), 'prototype_bbox.png'),
            epoch=start_epoch_number,
            index=j,
            bbox_height_start=prototype_info[j][1],
            bbox_height_end=prototype_info[j][2],
            bbox_width_start=prototype_info[j][3],
            bbox_width_end=prototype_info[j][4],
            color=(0, 255, 255)
        )

    # Find and save K nearest image patches to each prototype on the train set
    log('\nSaving nearest prototypes of train set')
    find_k_nearest_patches_to_prototypes(
        dataloader=train_loader,  # dataloader with images in [0,1]
        prototype_network_parallel=ppnet_multi,  # network exposing prototype_vectors
        k=args.top_imgs + 1,  # +1 to include the prototype’s own source image first
        preprocess_input_function=preprocess_input_function,  # apply normalization if required
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_train_images,
        log=log)
    # Same for the test set
    log('\nSaving nearest prototypes of test set')
    find_k_nearest_patches_to_prototypes(
        dataloader=test_loader,  # dataloader with images in [0,1]
        prototype_network_parallel=ppnet_multi,  # network exposing prototype_vectors
        k=args.top_imgs,
        preprocess_input_function=preprocess_input_function,  # apply normalization if required
        full_save=True,
        root_dir_for_saving_images=root_dir_for_saving_test_images,
        log=log)

    logclose() 

if __name__ == "__main__":
    import argparse
    from torch.serialization import add_safe_globals
    # Allowlist the classes pickled inside the checkpoint
    from ppnet.model import PPNet
    add_safe_globals([PPNet])
    try:
        from ppnet.resnet_features import ResNet_features
        add_safe_globals([ResNet_features])
    except Exception:
        try:
            from ppnet.vgg_features import VGG_features
            add_safe_globals([VGG_features])
    # If neither import works, just skip allowlisting (fallback)
        except Exception:
            pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--dataset", type=str, required=True)     # e.g. img/cub200
    parser.add_argument("--model", type=str, required=True)       # e.g. saved_models/.../300push_local.pth
    parser.add_argument("--out", type=str, default="runs")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--top_imgs", type=int, default=5)
    args = parser.parse_args()

    run_analysis(args)
