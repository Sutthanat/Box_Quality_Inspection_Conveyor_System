"""Post-it colour check on the CROPPED box image (same crop as used for YOLO).

Usage on a folder:      python postit_color.py crops_folder
Calibrate new colours:  python postit_color.py crops_folder --calibrate
In your Pi code:        from postit_color import detect_postit_color
                        name, info = detect_postit_color(crop_bgr)
"""
import cv2, sys, glob, numpy as np
from pathlib import Path

# OpenCV hue runs 0-180 and wraps around (red is both ~0 and ~180).
# Reference hues measured on YOUR post-its under YOUR lighting. Re-measure with --calibrate
# if lighting or post-it brand changes.
REFERENCE_HUES = {"orange": 10, "yellow": 38, "green": 49, "blue": 101, "pink": 165}
MAX_HUE_DISTANCE = 12      # farther than this from every reference -> "unknown"
S_MIN, V_MIN = 50, 170     # post-it is colourful AND bright (belt is dark, paper is colourless)
MIN_AREA = 1500            # pixels; smaller blobs are ignored

def hue_distance(a, b):
    d = abs(a - b) % 180
    return min(d, 180 - d)

def detect_postit_color(crop_bgr):
    """Returns (colour_name, info). colour_name is 'none' if no post-it blob is found
    (e.g. missing post-it), or 'unknown' if the colour matches no reference."""
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 1] > S_MIN) & (hsv[..., 2] > V_MIN)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n < 2 or stats[1:, cv2.CC_STAT_AREA].max() < MIN_AREA:
        return "none", {}
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sel = lab == k
    h, s, v = (float(np.median(hsv[..., i][sel])) for i in range(3))
    name, dist = min(((c, hue_distance(h, ref)) for c, ref in REFERENCE_HUES.items()), key=lambda t: t[1])
    if dist > MAX_HUE_DISTANCE:
        name = "unknown"
    x, y, w, hh = (int(stats[k, c]) for c in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return name, dict(hue=h, sat=s, val=v, area=int(stats[k, cv2.CC_STAT_AREA]), bbox=(x, y, w, hh))

if __name__ == "__main__":
    folder, calibrate = sys.argv[1], "--calibrate" in sys.argv
    files = [f for f in sorted(Path(folder).rglob("*")) if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    counts = {}
    for f in files:
        name, info = detect_postit_color(cv2.imread(str(f)))
        counts[name] = counts.get(name, 0) + 1
        extra = f"  hue={info['hue']:.0f} sat={info['sat']:.0f} val={info['val']:.0f}" if info else ""
        if calibrate or name in ("none", "unknown"):
            print(f"{f.name:40s}{name:9s}{extra}")
    print("\ncolour counts:", counts)
