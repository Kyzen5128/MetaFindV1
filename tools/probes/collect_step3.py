"""Every finished step-3 measurement into one table."""
import json, glob, os
C = ("text","image","pc","text+image","text+pc","image+pc","full")
PAPER = {"text":13.8,"image":11.7,"pc":75.1,"text+image":17.2,
         "text+pc":44.5,"image+pc":45.8,"full":51.7}
ULIP  = {"text":0.1,"image":0.1,"pc":97.9,"text+image":0.0,
         "text+pc":33.9,"image+pc":22.6,"full":6.4}

def row(label, cells):
    label = label if len(label) < 30 else label[:29]
    return label.ljust(30) + "".join(("%.1f" % (cells[c]["R@1"]*100)).rjust(9) for c in C)

print("=" * 86); print("MetaFind Stage 1 (pilot10b, 10 epochs) -- protocol D: 4,569 held-out vs 36,554")
print("=" * 86)
print("condition".ljust(30) + "".join(c.rjust(9) for c in C))
print("PAPER w/o ESSGNN".ljust(30) + "".join(("%.1f" % PAPER[c]).rjust(9) for c in C))
print("-" * 86)
seen = set()
for f in sorted(glob.glob("output/look/exp_*.json")):
    try: d = json.load(open(f))
    except Exception: continue
    r = d.get("results")
    if not isinstance(r, dict): continue
    for arm, cells in r.items():
        if not (isinstance(cells, dict) and "full" in cells and
                isinstance(cells.get("full"), dict) and "R@1" in cells["full"]):
            continue
        tag = os.path.basename(f)[4:-5]
        lbl = tag if arm == "full" else f"{tag}:{arm}"
        if lbl in seen: continue
        seen.add(lbl); print(row(lbl, cells))

print()
print("=" * 86); print("ULIP baseline -- released ULIP-2, no training, no fusion")
print("=" * 86)
print("condition".ljust(30) + "".join(c.rjust(9) for c in C))
print("PAPER ULIP".ljust(30) + "".join(("%.1f" % ULIP[c]).rjust(9) for c in C))
print("-" * 86)
for f in sorted(glob.glob("output/look/exp_ulip_*.json")):
    d = json.load(open(f))
    for arm, cells in d["results"].items():
        if arm == "modality_geometry" or "rawmean" not in arm: continue
        tag = os.path.basename(f)[9:-5]
        print(row(f"{tag}:{arm.split('__')[0]}", cells))
