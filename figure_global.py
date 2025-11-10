import os, argparse, glob
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def find_one(run_dir, j, suffix):
    pats = [
        os.path.join(run_dir, f"{j}_prototype-{suffix}"),
        os.path.join(run_dir, f"{j}-prototype-{suffix}"),
        os.path.join(run_dir, f"prototype-{j}-{suffix}"),
    ]
    for p in pats:
        # accetta .png, .jpg
        for ext in (".png", ".jpg", ".jpeg"):
            fp = p + ext
            if os.path.exists(fp):
                return fp
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="es. saved_models/.../img/epoch-300")
    ap.add_argument("--proto", type=int, required=True, help="id del prototipo (j)")
    ap.add_argument("--out", required=True, help="path immagine di output")
    args = ap.parse_args()

    j = args.proto
    rd = args.run_dir

    fp_orig  = find_one(rd, j, "img-original")
    fp_ovl   = find_one(rd, j, "img-original_with_self_act")
    fp_crop  = find_one(rd, j, "img")  # questo è tipicamente il crop della patch
    fp_actnp = os.path.join(rd, f"{j}_prototype-self-act.npy")

    if fp_orig is None and fp_crop is None:
        raise SystemExit(f"Non trovo artefatti per il prototipo {j} in {rd}")

    cols = []
    titles = []
    if fp_orig:
        cols.append(Image.open(fp_orig).convert("RGB")); titles.append("source (original)")
    if fp_ovl:
        cols.append(Image.open(fp_ovl).convert("RGB"));  titles.append("source + self-activation")
    if fp_crop:
        cols.append(Image.open(fp_crop).convert("RGB"));  titles.append("prototype patch")

    if fp_ovl is None and os.path.exists(fp_actnp) and fp_orig:
        import matplotlib.cm as cm
        base = Image.open(fp_orig).convert("RGB")
        A = np.load(fp_actnp)
        A = (A - A.min()) / (A.max() - A.min() + 1e-6)
        A = Image.fromarray((A*255).astype(np.uint8)).resize(base.size)
        A = A.convert("L")
        # colormap su alpha
        heat = Image.fromarray(np.uint8(cm.jet(np.array(A)) * 255))
        heat = heat.convert("RGBA")
        base_rgba = base.convert("RGBA")
        overlay = Image.alpha_composite(base_rgba, heat.putalpha(120))
        cols.insert(1, overlay.convert("RGB")); titles.insert(1, "source + self-activation")

    n = len(cols)
    plt.figure(figsize=(3.4*n, 3.8))
    for i, (im, t) in enumerate(zip(cols, titles), start=1):
        plt.subplot(1, n, i); plt.imshow(im); plt.axis("off"); plt.title(t, fontsize=10)
    plt.suptitle(f"Prototype {j}: global view", fontsize=12)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    plt.tight_layout(); plt.savefig(args.out, dpi=220); plt.close()
    print("Saved", args.out)

if __name__ == "__main__":
    main()
