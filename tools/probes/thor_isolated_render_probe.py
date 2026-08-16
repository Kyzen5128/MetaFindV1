"""Probe: can one ProcTHOR asset be rendered in isolation, comparable to n04?

Answer: yes. Recorded as F24. Kept because the answer took three attempts and
the working recipe is not obvious -- the house must be stripped of doors,
windows, walls and ceilings AND the asset lifted clear of the floor, or the
floor plane fills the frame and slices the object in half.

What this does NOT settle is point clouds. n03 samples a full mesh surface and
reaches occluded faces; eleven orbital depth frames reach only the visible hull.

Not a node, and deliberately not wired into the graph -- it answers a question
about U-08b and nothing downstream reads it. Run it by hand:

    python tools/probes/thor_isolated_render_probe.py

Needs ai2thor and a GPU; runs headless via CloudRendering and shares the card
with whatever else is running.
"""
import json, math
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering

house = json.loads(open('/mnt/data1/kyzen/MetaFind/datasets/procthor-10k/train.jsonl').readline())
solo = json.loads(json.dumps(house))
solo["objects"] = [{"assetId": "Fridge_19", "id": "Fridge|0|0",
                    "position": {"x": 4.0, "y": 40.0, "z": 3.0},
                    "rotation": {"x": 0, "y": 0, "z": 0}, "kinematic": True}]
solo["doors"] = []; solo["windows"] = []; solo["walls"] = []
for r in solo["rooms"]: r["ceilings"] = []

c = Controller(scene=solo, platform=CloudRendering, width=224, height=224,
               renderDepthImage=True, quality="Medium")
tgt = next(o for o in c.last_event.metadata["objects"] if o.get("assetId") == "Fridge_19")
bb = tgt["axisAlignedBoundingBox"]; ctr, size = bb["center"], bb["size"]
half = max(size.values()) * 0.62
r = 6.0

ok_ortho, frames, depths = False, [], []
for k in range(11):
    az = 360.0*k/11.0; rad = math.radians(az)
    pos = {"x": ctr["x"]+r*math.sin(rad), "y": ctr["y"]+r*0.34, "z": ctr["z"]+r*math.cos(rad)}
    rot = {"x": 20.0, "y": (az+180.0) % 360.0, "z": 0.0}
    try:
        ev = c.step(action="AddThirdPartyCamera", position=pos, rotation=rot,
                    orthographic=True, orthographicSize=half, skyboxColor="white")
        ok_ortho = ev.metadata["lastActionSuccess"]
    except Exception as e:
        print("orthographic rejected:", str(e)[:120]); break
    frames.append(ev.third_party_camera_frames[-1])
    d = ev.third_party_depth_frames[-1] if getattr(ev, "third_party_depth_frames", None) else None
    depths.append(d)

print("orthographic accepted:", ok_ortho, "| frames:", len(frames))
if frames:
    arr = np.stack(frames)[..., :3]
    bg = (arr > 240).all(axis=-1).mean(axis=(1,2))
    print("background fraction:", np.round(bg, 3))
    print("scale consistency (object pixel fraction) std:", round(float((1-bg).std()), 4))
    np.save(f'{__import__("os").path.dirname(__file__)}/thor_ortho.npy', arr)
if depths and depths[0] is not None:
    D = np.stack([d for d in depths if d is not None])
    print("depth frames:", D.shape, "| finite:", float(np.isfinite(D).mean()))
    near = D[np.isfinite(D)]
    print("depth range:", round(float(near.min()),2), "-", round(float(near.max()),2))
else:
    print("no third-party depth frames returned")
c.stop()
