import os, torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

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

from ppnet.preprocess import mean, std

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True, help="checkpoint .pth")
parser.add_argument("--dataset", required=True, help="root with train/ and test/ (e.g. img/cub200)")
parser.add_argument("--gpus", default="0")
parser.add_argument("--batch", type=int, default=128)
parser.add_argument("--num_workers", type=int, default=4)
args = parser.parse_args()

if args.gpus is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load model (checkpoint stores the full object, not only state_dict)
ppnet = torch.load(args.model, weights_only=False, map_location=device)
ppnet.eval()
if torch.cuda.is_available():
    ppnet = ppnet.cuda()
    net = torch.nn.DataParallel(ppnet)
else:
    net = ppnet

# image size from model
img_size = net.module.img_size if hasattr(net, "module") else net.img_size

# test loader (WITH normalization)
tf = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std),
])
test_dir = os.path.join(args.dataset, "test")
test_ds = datasets.ImageFolder(test_dir, tf)
test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                         num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

# evaluate
correct = 0
total = 0
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = net(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

acc = 100.0 * correct / total
print(f"Top-1 Accuracy: {acc:.2f}%  (correct={correct}/{total})")
