import os, argparse, torch
from torch.serialization import add_safe_globals
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

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

def get_img_size(net):
    return net.module.img_size if hasattr(net, "module") else net.img_size

@torch.no_grad()
def top1_acc(net, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = net(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / total

@torch.no_grad()
def delta_acc_at_k(net, loader, device, k, max_samples=None):
    """
    Per-image masking: for each image (batch_size=1), find its top-k prototypes
    by minimal distance; then zero the corresponding columns in the last layer
    ONLY for that forward, recompute logits, and track accuracy.
    """
    was_training = net.training
    net.eval()

    # convenience: handle DataParallel
    module = net.module if hasattr(net, "module") else net
    last_w = module.last_layer.weight.data.clone()  # [num_classes, num_prototypes]
    max_dist = (module.prototype_shape[1] * module.prototype_shape[2] * module.prototype_shape[3])

    correct_orig = correct_mask = total = 0

    count = 0
    for x, y in loader:
        if max_samples is not None and count >= max_samples:
            break
        count += 1

        x, y = x.to(device), y.to(device)
        # original
        logits, min_d = net(x)  # min_d shape: [1, n_prototypes, H, W]
        pred_orig = logits.argmax(1)
        correct_orig += (pred_orig == y).sum().item()

        # get per-prototype minimal distance (smaller = more activated)
        d = min_d.view(min_d.size(0), min_d.size(1), -1).min(-1).values  # [1, n_prototypes]
        topk_idx = torch.topk(-d, k, dim=1).indices.squeeze(0)  # largest -d == smallest distance

        # mask those prototypes (zero their contribution)
        masked_w = last_w.clone()
        masked_w[:, topk_idx] = 0.0
        # swap weights temporarily
        orig_w = module.last_layer.weight.data
        module.last_layer.weight.data = masked_w
        try:
            logits_masked, _ = net(x)
        finally:
            # restore weights
            module.last_layer.weight.data = orig_w

        pred_mask = logits_masked.argmax(1)
        correct_mask += (pred_mask == y).sum().item()
        total += y.size(0)

    if was_training: net.train()

    acc_orig = 100.0 * correct_orig / total
    acc_mask = 100.0 * correct_mask / total
    delta = acc_orig - acc_mask
    return acc_orig, acc_mask, delta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True, help="root with train/ and test/")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--k", type=int, nargs="+", default=[1,3,5])
    ap.add_argument("--samples", type=int, default=1000, help="test images to sample for speed (set -1 for all)")
    ap.add_argument("--batch", type=int, default=1, help="must be 1 for per-image masking; leave as 1")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load model
    net = torch.load(args.model, weights_only=False, map_location=device)
    net.eval()
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).to(device)

    # data
    img_size = get_img_size(net)
    tf = transforms = __import__("torchvision.transforms").transforms
    tt = tf.Compose([
        tf.Resize((img_size, img_size)),
        tf.ToTensor(),
        tf.Normalize(mean=mean, std=std),
    ])
    test_ds = datasets.ImageFolder(os.path.join(args.dataset, "test"), tt)

    # subsample for speed if requested
    if args.samples is not None and args.samples > 0 and args.samples < len(test_ds):
        idx = np.random.RandomState(0).choice(len(test_ds), size=args.samples, replace=False)
        test_ds = Subset(test_ds, idx)

    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
                             num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    # baseline on the same subset
    baseline = top1_acc(net, test_loader, device)
    print(f"Baseline Top-1 on subset: {baseline:.2f}%")

    # compute ΔAcc@k
    for k in args.k:
        acc_orig, acc_mask, delta = delta_acc_at_k(net, test_loader, device, k, max_samples=None)
        print(f"k={k}: orig={acc_orig:.2f}%  masked={acc_mask:.2f}%  ΔAcc={delta:.2f} pp")

if __name__ == "__main__":
    main()
