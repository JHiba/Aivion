import os
import argparse
import torch
from PIL import Image
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

def run_worker():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str)
    parser.add_argument("--images", nargs="+")
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Load Pipeline to GPU
    pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipe.to("cuda")

    if args.images:
        print(f"📦 Processing {len(args.images)} views...")
        imgs = [Image.open(f).convert("RGB") for f in args.images[:7]]
        outputs = pipe.run_multi_image(imgs, formats=["gaussian", "mesh"])
    else:
        img = Image.open(args.image).convert("RGB")
        outputs = pipe.run(img, formats=["gaussian", "mesh"])
    
    # postprocessing logic matching trellis_ui.py
    glb = postprocessing_utils.to_glb(outputs["gaussian"][0], outputs["mesh"][0], simplify=0.95, texture_size=1024)
    glb.export(args.output)
    print(f"✅ Exported to {args.output}")

if __name__ == "__main__":
    run_worker()
