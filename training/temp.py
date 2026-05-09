"""
prepare_custom_dataset.py
--------------------------
Scans your 3 folders (google_images, State-wise_OLX, video_images)
Converts Pascal VOC XML → YOLO format
Writes combined_dataset/data.yaml ready for training

NO Roboflow — your custom dataset only.

USAGE:
  python prepare_custom_dataset.py
"""

import os, shutil, random
import xml.etree.ElementTree as ET
from pathlib import Path

# ─── EDIT THESE 2 PATHS ───────────────────────────────────────────────────────
DATASET_ROOT  = r"C:\Users\HP\Desktop\Projects\Number plates dataset"
OUTPUT_FOLDER = r"C:\Users\HP\Desktop\Projects\License_PD\combined_dataset"
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_SPLIT = 0.85
RANDOM_SEED = 42
IMAGE_EXTS  = {".jpg", ".jpeg", ".png", ".bmp"}


def voc_to_yolo(xmin, ymin, xmax, ymax, img_w, img_h):
    xmin, ymin = max(0, xmin), max(0, ymin)
    xmax, ymax = min(img_w, xmax), min(img_h, ymax)
    if xmax <= xmin or ymax <= ymin:
        return None
    cx = ((xmin + xmax) / 2) / img_w
    cy = ((ymin + ymax) / 2) / img_h
    bw = (xmax - xmin) / img_w
    bh = (ymax - ymin) / img_h
    return round(cx,6), round(cy,6), round(bw,6), round(bh,6)


def parse_xml(xml_path):
    try:
        root  = ET.parse(xml_path).getroot()
        size  = root.find("size")
        img_w = int(float(size.find("width").text))
        img_h = int(float(size.find("height").text))
        if img_w == 0 or img_h == 0:
            return None, []
        fn    = root.find("filename")
        fname = fn.text if fn is not None else None
        lines = []
        for obj in root.findall("object"):
            bb = obj.find("bndbox")
            try:
                coords = (int(float(bb.find("xmin").text)),
                          int(float(bb.find("ymin").text)),
                          int(float(bb.find("xmax").text)),
                          int(float(bb.find("ymax").text)))
            except Exception:
                continue
            r = voc_to_yolo(*coords, img_w, img_h)
            if r:
                lines.append(f"0 {r[0]} {r[1]} {r[2]} {r[3]}")
        return fname, lines
    except Exception:
        return None, []


def find_image(xml_path, xml_fname):
    folder = os.path.dirname(xml_path)
    base   = os.path.splitext(os.path.basename(xml_path))[0]
    for ext in IMAGE_EXTS:
        p = os.path.join(folder, base + ext)
        if os.path.exists(p): return p
        p = os.path.join(folder, base + ext.upper())
        if os.path.exists(p): return p
    if xml_fname:
        for name in [xml_fname, os.path.basename(xml_fname)]:
            p = os.path.join(folder, name)
            if os.path.exists(p): return p
    return None


def scan(folder, label):
    samples, skipped = [], 0
    xmls = list(Path(folder).rglob("*.xml"))
    print(f"  [{label}] {len(xmls)} XMLs ...", end=" ", flush=True)
    for xml_path in xmls:
        fname, lines = parse_xml(str(xml_path))
        if not lines: skipped += 1; continue
        img = find_image(str(xml_path), fname)
        if not img: skipped += 1; continue
        samples.append((img, lines))
    print(f"✓ {len(samples)} valid  ✗ {skipped} skipped")
    return samples


def main():
    print("=" * 60)
    print("  Custom Indian Plate Dataset Builder")
    print("=" * 60)

    all_samples = []

    print("\n[1/3] Scanning folders ...")

    g = os.path.join(DATASET_ROOT, "google_images")
    if os.path.exists(g):
        all_samples += scan(g, "google_images")
    else:
        print(f"  [SKIP] google_images not found")

    sw = os.path.join(DATASET_ROOT, "State-wise_OLX")
    if os.path.exists(sw):
        states = sorted(d for d in os.listdir(sw)
                        if os.path.isdir(os.path.join(sw, d)))
        state_samples = []
        for state in states:
            state_samples += scan(os.path.join(sw, state), f"OLX/{state}")
        print(f"  → State-wise subtotal: {len(state_samples)} across {len(states)} states")
        all_samples += state_samples
    else:
        print(f"  [SKIP] State-wise_OLX not found")

    v = os.path.join(DATASET_ROOT, "video_images")
    if os.path.exists(v):
        all_samples += scan(v, "video_images")
    else:
        print(f"  [SKIP] video_images not found")

    if not all_samples:
        print("\n[ERROR] No images found. Check DATASET_ROOT path.")
        return

    print(f"\n  Total collected: {len(all_samples)} images")

    print(f"\n[2/3] Shuffling and splitting ...")
    random.seed(RANDOM_SEED)
    random.shuffle(all_samples)
    n_train   = int(len(all_samples) * TRAIN_SPLIT)
    train_set = all_samples[:n_train]
    val_set   = all_samples[n_train:]
    print(f"  Train: {len(train_set)} | Val: {len(val_set)}")

    print(f"\n[3/3] Writing dataset to: {OUTPUT_FOLDER} ...")
    for split in ["train", "val"]:
        os.makedirs(os.path.join(OUTPUT_FOLDER, "images", split), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_FOLDER, "labels", split), exist_ok=True)

    def write(samples, split):
        for i, (src, lines) in enumerate(samples):
            ext     = os.path.splitext(src)[1].lower() or ".jpg"
            dst_img = os.path.join(OUTPUT_FOLDER, "images", split, f"{split}_{i:06d}{ext}")
            dst_lbl = os.path.join(OUTPUT_FOLDER, "labels", split, f"{split}_{i:06d}.txt")
            try:
                shutil.copy2(src, dst_img)
                with open(dst_lbl, "w") as f:
                    f.write("\n".join(lines))
            except Exception as e:
                print(f"  [WARN] {os.path.basename(src)}: {e}")

    write(train_set, "train")
    write(val_set,   "val")

    out_fwd = OUTPUT_FOLDER.replace("\\", "/")
    yaml = f"""# Custom Indian License Plate Dataset — All 35 States
path: {out_fwd}
train: images/train
val:   images/val
nc: 1
names: ['License_Plate']
"""
    yaml_path = os.path.join(OUTPUT_FOLDER, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml)

    print("\n" + "=" * 60)
    print("✅  Done!")
    print(f"   Train  : {len(train_set)}")
    print(f"   Val    : {len(val_set)}")
    print(f"   YAML   : {yaml_path}")
    print("\n  Next → run train.py")
    print("=" * 60)


if __name__ == "__main__":
    main()