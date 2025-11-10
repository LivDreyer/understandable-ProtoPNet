# eval_consistency.py
import os, argparse, torch
from torch.serialization import add_safe_globals
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from collections import defaultdict

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
def topk_prototypes(net, loader, device, k):
    """
    Returns: list of (class_id, set_of_topk_proto_idx) for each image.
    """
    net.eval()
    out = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits, min_d = net(x)  # [B, P, H, W]
        # per-prototype minimal distance
        d = min_d.view(min_d.size(0), min_d.size(1), -1).min(-1).values  # [B, P]
        topk = torch.topk(-d, k, dim=1).indices.cpu().numpy()  # larger -d => smaller distance
        for i in range(x.size(0)):
            out.append((int(y[i].item()), set(topk[i].tolist())))
    return out

def jaccard(a, b):
    u = len(a | b)
    if u == 0: return 0.0
    return len(a & b) / u

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--k", type=int, nargs="+", default=[1,3,5])
    ap.add_argument("--samples_per_class", type=int, default=25,
                    help="number of test images per class to sample for pairwise Jaccard")
    ap.add_argument("--num_workers", type=int, default=2)
    args = ap.parse_args()

    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net = torch.load(args.model, weights_only=False, map_location=device)
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net).to(device)

    img_size = get_img_size(net)
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    full_test = datasets.ImageFolder(os.path.join(args.dataset, "test"), tf)

    # Build a balanced subset: up to N images per class
    by_class = defaultdict(list)
    for i, (_, y) in enumerate(full_test.samples):
        by_class[y].append(i)
    sel_idx = []
    rng = np.random.default_rng(0)
    for cls, idxs in by_class.items():
        take = min(args.samples_per_class, len(idxs))
        sel_idx.extend(rng.choice(idxs, size=take, replace=False).tolist())
    subset = Subset(full_test, sel_idx)

    loader = DataLoader(subset, batch_size=64, shuffle=False,
                        num_workers=args.num_workers, pin_memory=torch.cuda.is_available())

    for k in args.k:
        tuples = topk_prototypes(net, loader, device, k)  # list of (cls, set)
        # group by class
        per_cls_sets = defaultdict(list)
        for c, s in tuples:
            per_cls_sets[c].append(s)
        # pairwise jaccard within class
        js = []
        for c, sets in per_cls_sets.items():
            n = len(sets)
            if n < 2: continue
            for i in range(n):
                for j in range(i+1, n):
                    js.append(jaccard(sets[i], sets[j]))
        mean_j = float(np.mean(js)) if len(js) else 0.0
        print(f"Jaccard@{k} (same-class mean): {mean_j:.4f}  (pairs={len(js)})")

if __name__ == "__main__":
    main()



