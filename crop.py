"""Fixed-window crop for the box-inspection dataset (box at 0 degrees).
Usage:  python crop.py <raw_root_folder> <output_root_folder>
Every image is cropped with the SAME window. Sub-folder names and file names are kept.
The Pi must use exactly the same window (X0,Y0,X1,Y1) before running the model."""
import cv2, sys
from pathlib import Path

X0, Y0, X1, Y1 = 535, 80, 855, 700     # pixels in the 1280x720 photo (belt + room for open lid)
EXTS = {".jpg", ".jpeg", ".png"}

def crop_fixed(img):
    return img[Y0:Y1, X0:X1]

def main(src, dst):
    src, dst = Path(src), Path(dst)
    n = 0
    for f in sorted(src.rglob("*")):
        if f.suffix.lower() not in EXTS: continue
        img = cv2.imread(str(f))
        if img is None or img.shape[:2] != (720, 1280):
            print("skipped (unexpected size):", f); continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop_fixed(img), [cv2.IMWRITE_JPEG_QUALITY, 98])
        n += 1
    print(f"cropped {n} images -> {dst}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
