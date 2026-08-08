#!/usr/bin/env python3
"""
Minimal SAM3 zero-shot open-vocabulary detector for traffic signs and
traffic lights, for tiles that have no pre-existing detections.jsonl.

The original part15 detections came from a script of this same name
referenced throughout this project's documentation (bucket_signs_sam3.py's
docstring, HOVER_LABELS_STATUS.md) as the source of the "sam3_score"-tagged
detections.jsonl -- but that script no longer exists anywhere in this
project. This is a from-scratch rebuild, matching the same SAM3 calling
pattern already proven to work in bucket_signs_sam3.py (Sam3VideoModel /
Sam3VideoProcessor, one prompt per session -- multi-prompt-per-session was
found there to break SAM3's score calibration), but doing initial
DETECTION on full frames (two broad prompts: "traffic light", "traffic
sign") rather than re-classification of an already-known crop into one of
76 StVO classes.

Output schema matches the original detections.jsonl exactly, so it is a
drop-in input to colorize_pointcloud.py's localize_detections() and the
rest of the existing pipeline:
    {"image": ..., "kind": "light"|"sign", "bbox_px": [x1,y1,x2,y2], "sam3_score": ...}

bbox_px stays in the same pixel frame as the phase0-corrected image on
disk (undistorted, but NOT rotated for "left"/"right" cameras' sideways
storage) -- matching how the existing part15 detections.jsonl was
evidently produced (localize_detections() already assumes this frame), not
the upright-corrected frame bucket_signs_sam3.py's load_corrected_crop()
produces for classification display purposes only.

Usage:
    python detect_signs_lights_sam3.py <laz_file> [--limit N]
"""

import sys
import os
import json
import time
import signal
import argparse

import numpy as np
import torch
from PIL import Image

DATA_FIX_DIR = "/home/rohit/Documents/Output/data_fix"
CAMEXTR_PATH = "/home/rohit/Downloads/Project/hd_map_p06/config/2026_05_08_Bonn/CamExtr.json"
SAM3_MODEL_ID = "facebook/sam3"

PROMPTS = {"light": "traffic light", "sign": "traffic sign"}
INTERNAL_SCORE_THRESHOLD = 0.3   # model's own gate (this IS the detection task, unlike bucketing)
KEEP_SCORE_FLOOR = 0.5           # our own gate on top, applied when filtering the output
MAX_IMG_DIM_PX = 1024            # resize cap for detection speed/memory; boxes rescaled back to original size
PER_PROMPT_TIMEOUT_S = 20
BBOX_MARGIN_M = 20.0


class DetTimeout(Exception):
    pass


def _timeout_handler(signum, frame):
    raise DetTimeout()


def get_laz_bbox(laz_path, margin=BBOX_MARGIN_M):
    import laspy
    with laspy.open(laz_path) as f:
        hdr = f.header
        mins, maxs = hdr.mins, hdr.maxs
    return {
        "e_min": mins[0] - margin, "e_max": maxs[0] + margin,
        "n_min": mins[1] - margin, "n_max": maxs[1] + margin,
    }


def load_and_filter_camextr(bbox):
    with open(CAMEXTR_PATH) as f:
        data = json.load(f)
    entries = data.get("Profiler_0", [])
    filtered = [e for e in entries if bbox["e_min"] <= e["Xyz"][0] <= bbox["e_max"]
                and bbox["n_min"] <= e["Xyz"][1] <= bbox["n_max"]]
    return filtered


def detect_one_prompt(processor, model, device, img, prompt_text):
    """One prompt, its own SAM3 session. Returns list of (box_xyxy, score)
    in the (possibly resized) image's own pixel frame, or [] on timeout/OOM."""
    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(PER_PROMPT_TIMEOUT_S)
    try:
        session = processor.init_video_session(video=[img], inference_device=device, dtype=torch.bfloat16)
        processor.add_text_prompt(session, prompt_text)
        with torch.inference_mode():
            outputs = model(inference_session=session, frame_idx=0)
        result = processor.postprocess_outputs(session, outputs, original_sizes=[[img.height, img.width]])
    except DetTimeout:
        return []
    except torch.cuda.OutOfMemoryError:
        return []
    finally:
        signal.alarm(0)
        if device == "cuda":
            torch.cuda.empty_cache()

    obj_ids = result["prompt_to_obj_ids"].get(prompt_text, [])
    if not obj_ids:
        return []
    id_to_idx = {oid: i for i, oid in enumerate(result["object_ids"].tolist())}
    boxes = result["boxes"]
    scores = result["scores"]
    out = []
    for oid in obj_ids:
        idx = id_to_idx[oid]
        out.append((boxes[idx].tolist(), float(scores[idx])))
    return out


def process_image(processor, model, device, img_path):
    img = Image.open(img_path).convert("RGB")
    orig_w, orig_h = img.size
    longest = max(orig_w, orig_h)
    scale = 1.0
    if longest > MAX_IMG_DIM_PX:
        scale = MAX_IMG_DIM_PX / longest
        img_small = img.resize((max(1, round(orig_w * scale)), max(1, round(orig_h * scale))), Image.LANCZOS)
    else:
        img_small = img

    results = {}
    for kind, prompt_text in PROMPTS.items():
        dets = detect_one_prompt(processor, model, device, img_small, prompt_text)
        results[kind] = dets
    return results, scale


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("laz_path")
    ap.add_argument("--limit", type=int, default=None, help="process only first N images (smoke test)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = args.out or os.path.join(
        os.path.dirname(args.laz_path),
        os.path.basename(args.laz_path).replace(".laz", "_detections_sam3.jsonl"))

    print(f"[1/3] Reading LAZ bbox + filtering CamExtr...", flush=True)
    bbox = get_laz_bbox(args.laz_path)
    entries = load_and_filter_camextr(bbox)
    if args.limit:
        entries = entries[:args.limit]
    print(f"  {len(entries)} camera entries to process", flush=True)

    print("[2/3] Loading SAM3...", flush=True)
    from transformers import Sam3VideoModel, Sam3VideoProcessor
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
    model = Sam3VideoModel.from_pretrained(
        SAM3_MODEL_ID, dtype=torch.bfloat16, score_threshold_detection=INTERNAL_SCORE_THRESHOLD,
    ).to(device).eval()
    print(f"  Loaded on {device}", flush=True)

    print(f"[3/3] Detecting on {len(entries)} images (2 prompts each)...", flush=True)
    t0 = time.time()
    n_written = 0
    n_missing = 0
    n_with_det = 0
    with open(out_path, "w") as f:
        for i, entry in enumerate(entries):
            filename = entry["Image"]
            img_path = os.path.join(DATA_FIX_DIR, filename)
            if not os.path.isfile(img_path):
                n_missing += 1
                continue

            results, scale = process_image(processor, model, device, img_path)
            got_any = False
            for kind, dets in results.items():
                for box, score in dets:
                    if score < KEEP_SCORE_FLOOR:
                        continue
                    # rescale box back to the ORIGINAL (phase0-corrected) image's own pixel frame
                    box_orig = [c / scale for c in box]
                    f.write(json.dumps({
                        "image": filename, "kind": kind,
                        "bbox_px": [round(v, 1) for v in box_orig],
                        "sam3_score": round(score, 4),
                    }) + "\n")
                    n_written += 1
                    got_any = True
            if got_any:
                n_with_det += 1
            f.flush()

            if (i + 1) % 25 == 0 or (i + 1) == len(entries):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(entries) - (i + 1)) / rate if rate > 0 else float("inf")
                print(f"    [{i+1}/{len(entries)}] detections={n_written} "
                      f"images_with_det={n_with_det} missing={n_missing} "
                      f"({rate:.2f} img/s, ETA {eta/60:.1f} min)", flush=True)

    print(f"\nDone: {n_written} detections across {n_with_det} images "
          f"({len(entries)} images checked, {n_missing} missing)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
