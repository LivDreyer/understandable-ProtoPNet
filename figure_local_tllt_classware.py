import os, glob, argparse, re
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# -------- utilities --------
def resolve_test_image(test_arg, roots):
    if os.path.exists(test_arg):
        return os.path.abspath(test_arg)
    base = os.path.basename(test_arg)
    if not base.lower().endswith(('.jpg','.jpeg','.png')):
        base += '.jpg'
    for root in roots:
        for dp,_,files in os.walk(root):
            if base in files:
                return os.path.abspath(os.path.join(dp, base))
    raise FileNotFoundError(f"Immagine non trovata: {test_arg}")

def load_images_txt(path):
    id2rel = {}
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                i, rel = line.strip().split()
                id2rel[int(i)] = rel
    return id2rel

def prefer_proto_render(run_dir, j):
    # overlay > crop > original
    for suf in ["img-original_with_self_act", "img", "img-original"]:
        for ext in (".png",".jpg",".jpeg"):
            p = os.path.join(run_dir, f"{j}_prototype-{suf}{ext}")
            if os.path.exists(p):
                return p
    return None

def try_load_last_layer_row(ckpt_path, class_id_zero_based):
    """
    Prova a leggere last_layer.weight[C,P] dal checkpoint.
    Ritorna un vettore di lunghezza P (float) oppure None.
    """
    try:
        import torch
        blob = torch.load(ckpt_path, map_location="cpu")
        # vari layout possibili
        if isinstance(blob, dict):
            # alcuni salvano direttamente lo state_dict
            sd = blob.get("model_state_dict") or blob.get("state_dict") or blob
        else:
            return None
        # trova una chiave che finisca con 'last_layer.weight'
        key = None
        for k in sd.keys():
            if k.endswith("last_layer.weight"):
                key = k; break
        if key is None:
            return None
        W = sd[key]  # shape [C, P]
        W = W.cpu().numpy()
        if class_id_zero_based < 0 or class_id_zero_based >= W.shape[0]:
            return None
        return W[class_id_zero_based]  # (P,)
    except Exception:
        return None

def draw_panel(test_img, proto_list, out_path, title="Local “This-Looks-Like-That”"):
    k = len(proto_list)
    plt.figure(figsize=(3.2*(k+1), 3.6))
    im = Image.open(test_img).convert("RGB")
    plt.subplot(1, k+1, 1); plt.imshow(im); plt.axis('off'); plt.title("test image", fontsize=9)
    for i,(j,p,score_txt) in enumerate(proto_list, start=2):
        im = Image.open(p).convert("RGB")
        plt.subplot(1, k+1, i); plt.imshow(im); plt.axis('off'); plt.title(f"proto {j}\n{score_txt}", fontsize=8)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.suptitle(title, fontsize=12)
    plt.tight_layout(); plt.savefig(out_path, dpi=220); plt.close()
    print("Saved:", out_path)

# -------- main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help=".../img/epoch-XXX/")
    ap.add_argument("--bb", default="saved_models/resnet34/debug_gpu/img/epoch-300/bb.npy",
                    help="path a bb.npy (proto_id -> image_id)")
    ap.add_argument("--images_txt", default="img/cub200/CUB_200_2011/images.txt",
                    help="CUB images.txt (image_id -> relative_path)")
    ap.add_argument("--test", required=True, help="path/filename o numero (es. 010.Red_winged_Blackbird/487.jpg)")
    ap.add_argument("--out", default="figures/fig2_local_tllt.png")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--ckpt", default="saved_models/resnet34/debug_gpu/checkpoints/300push_local.pth",
                    help="checkpoint per leggere i pesi last_layer")
    ap.add_argument("--roots", nargs="*", default=[
        "img/cub200/test", "img/cub200/train",
        "img/CUB_200_2011/original/CUB_200_2011/images",
        "img/cub200/CUB_200_2011/images",
    ])
    args = ap.parse_args()

    # 1) test image + class id (da prefisso numerico '010.' -> 9 come index 0-based)
    test_img = resolve_test_image(args.test, args.roots)
    test_class_dir = os.path.basename(os.path.dirname(test_img))  # "010.Red_winged_Blackbird"
    class_prefix = re.match(r"^(\d+)\.", test_class_dir)
    class_id_0 = int(class_prefix.group(1)) - 1 if class_prefix else None

    # 2) mapping
    bb = np.load(args.bb, allow_pickle=True)
    id2rel = load_images_txt(args.images_txt)

    # 3) pesi della last layer per la classe (se disponibili)
    last_row = try_load_last_layer_row(args.ckpt, class_id_0) if class_id_0 is not None else None

    # 4) seleziona prototipi della stessa classe della test image
    j_same = []
    for j in range(bb.size):
        img_id = int(bb.flat[j])
        rel = id2rel.get(img_id)
        if not rel: continue
        proto_class = os.path.dirname(rel)  # "010.Red_winged_Blackbird"
        if proto_class == test_class_dir:
            img_path = prefer_proto_render(args.run_dir, j)
            if img_path:
                score_txt = "w=n/a"
                if last_row is not None and j < last_row.shape[0]:
                    score_txt = f"w={last_row[j]:.2f}"
                j_same.append((j, img_path, score_txt))

    # 5) fallback se meno di K
    if len(j_same) < args.k:
        found = set(j for j,_,_ in j_same)
        extras = []
        for fp in glob.glob(os.path.join(args.run_dir, "*_prototype-*.png")):
            m = re.search(r"[/\\](\d+)_prototype-", fp)
            if not m: continue
            j = int(m.group(1))
            if j in found: continue
            score_txt = "w=n/a"
            if last_row is not None and j < last_row.shape[0]:
                score_txt = f"w={last_row[j]:.2f}"
            extras.append((j, fp, score_txt))
            if len(j_same)+len(extras) >= args.k: break
        j_same += extras

    proto_list = j_same[:args.k]
    if not proto_list:
        raise SystemExit("Nessun prototipo disponibile per la figura.")

    # 6) disegna
    draw_panel(test_img, proto_list, args.out)

if __name__ == "__main__":
    main()
