# push prototypes and generate img/epoch-300/bb.npy

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# config
MODEL     = 'saved_models/resnet34/debug_gpu/checkpoints/300nopush77.24.pth'
TRAIN_DIR = 'img/cub200/train'
OUT_DIR   = 'saved_models/resnet34/debug_gpu/img'   # global_analysis.py expects .../img/epoch-300/
BATCH     = 64
NUM_WORKERS = 4
EPOCH     = 300

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

from ppnet.push import push_prototypes
from ppnet.preprocess import preprocess_input_function

# device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device} | torch.cuda.is_available()={torch.cuda.is_available()}')

# load model
# weights_only=False because the file stores the full model object, not a state_dict
ppnet = torch.load(MODEL, weights_only=False, map_location=device)
ppnet.to(device)

# Use DataParallel if CUDA is available
ppnet_multi = torch.nn.DataParallel(ppnet) if torch.cuda.is_available() else ppnet

img_size = ppnet.module.img_size if hasattr(ppnet, 'module') else ppnet.img_size
train_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])
train_ds = datasets.ImageFolder(TRAIN_DIR, train_tf)
train_loader = DataLoader(
    train_ds,
    batch_size=BATCH,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

proto_prefix = 'prototype-img'       # -> {j}_prototype-img-original.png
act_prefix   = 'prototype-self-act'
bb_prefix    = 'bb'                  # -> bb.npy

push_prototypes(
    dataloader=train_loader,
    prototype_network_parallel=ppnet_multi,
    class_specific=True,
    preprocess_input_function=preprocess_input_function,
    prototype_layer_stride=1,
    root_dir_for_saving_prototypes=OUT_DIR,
    epoch_number=EPOCH,
    prototype_img_filename_prefix=proto_prefix,
    prototype_self_act_filename_prefix=act_prefix,
    proto_bound_boxes_filename_prefix=bb_prefix,
    save_prototype_class_identity=True,
    log=print,
    prototype_activation_function_in_numpy=None
)

pushed_ckpt = os.path.join(os.path.dirname(MODEL), '300push_local.pth')
to_save = ppnet.module if hasattr(ppnet, 'module') else ppnet
torch.save(to_save, pushed_ckpt)

print(f'\nSaved pushed checkpoint to: {pushed_ckpt}')
bb_path = os.path.join(OUT_DIR, f'epoch-{EPOCH}', 'bb.npy')
print('bb.npy exists:', os.path.exists(bb_path), '|', bb_path)
