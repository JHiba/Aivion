import os
import io
import time
import tempfile
import zipfile
from typing import Any, Optional, Tuple, List

import gradio as gr
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove, new_session
from scipy.ndimage import gaussian_filter

import torch
from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils


# =========================
#  Rembg model sessions
# =========================
session_u2net = new_session("u2net")
session_isnet = new_session("isnet-general-use")
session_silueta = new_session("silueta")


# =========================
#  Utility functions
# =========================

#used when user selects solid bg
def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    """Parse hex or rgb/rgba string to RGBA tuple."""
    if not color_str:
        return (255, 255, 255, 255)
    s = color_str.strip()

    # Hex
    if s.startswith("#"):
        s = s.lstrip("#")
        if len(s) == 6:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            return (r, g, b, 255)
        if len(s) == 8:
            r = int(s[0:2], 16)
            g = int(s[2:4], 16)
            b = int(s[4:6], 16)
            a = int(s[6:8], 16)
            return (r, g, b, a)

    # rgb / rgba
    if s.lower().startswith("rgb"):
        inside = s[s.find("(") + 1 : s.rfind(")")]
        parts = [p.strip() for p in inside.split(",")]
        try:
            r = int(float(parts[0]))
            g = int(float(parts[1]))
            b = int(float(parts[2]))
            if len(parts) == 4:
                a = float(parts[3])
                a = int(a * 255) if a <= 1 else int(a)
            else:
                a = 255
            return (r, g, b, a)
        except Exception:
            return (255, 255, 255, 255)

    return (255, 255, 255, 255)

#image enhancements
def _auto_white_balance_gray_world(img_rgb: np.ndarray) -> np.ndarray:
    eps = 1e-6
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    mean_r = r.mean() + eps
    mean_g = g.mean() + eps
    mean_b = b.mean() + eps
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    scale_r = mean_gray / mean_r
    scale_g = mean_gray / mean_g
    scale_b = mean_gray / mean_b

    out = img_rgb.astype(np.float32).copy()
    out[..., 0] = np.clip(out[..., 0] * scale_r, 0, 255)
    out[..., 1] = np.clip(out[..., 1] * scale_g, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * scale_b, 0, 255)
    return out.astype(np.uint8)


def _apply_temperature(img_rgb: np.ndarray, temp_value: int) -> np.ndarray:
    if temp_value == 0:
        return img_rgb
    max_delta = 0.10
    delta = (temp_value / 50.0) * max_delta
    r_scale = 1.0 + delta
    b_scale = 1.0 - delta

    out = img_rgb.astype(np.float32).copy()
    out[..., 0] = np.clip(out[..., 0] * r_scale, 0, 255)
    out[..., 2] = np.clip(out[..., 2] * b_scale, 0, 255)
    return out.astype(np.uint8)


def enhanced_preprocess_image(
    image: Image.Image,
    do_resize: bool = True,
    brightness: float = 1.0,
    sharpness: float = 1.0,
    saturation: float = 1.0,
    auto_wb: bool = False,
    temperature: int = 0,
) -> Image.Image:
    img = image.convert("RGBA")
    rgb = img.convert("RGB")

    if auto_wb:
        arr = np.array(rgb)
        arr = _auto_white_balance_gray_world(arr)
        rgb = Image.fromarray(arr, mode="RGB")

    if temperature != 0:
        arr = np.array(rgb)
        arr = _apply_temperature(arr, temperature)
        rgb = Image.fromarray(arr, mode="RGB")

    if saturation != 1.0:
        rgb = ImageEnhance.Color(rgb).enhance(float(saturation))
    if brightness != 1.0:
        rgb = ImageEnhance.Brightness(rgb).enhance(float(brightness))
    if sharpness != 1.0:
        rgb = ImageEnhance.Sharpness(rgb).enhance(float(sharpness))

    img = rgb.convert("RGBA")
    img = ImageEnhance.Contrast(img).enhance(1.15)

    if do_resize and max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return img


#I background removal often leaves "pixelated" or jagged edges. this creates a soft transition
def smooth_edges_only(image: Image.Image, blur_amount: float = 1.0) -> Image.Image:
    if blur_amount == 0:
        return image
    img = image.convert("RGBA")
    arr = np.array(img)
    if arr.shape[2] < 4:
        return image
    alpha = arr[:, :, 3].astype(np.float32)
    alpha_blurred = gaussian_filter(alpha, sigma=blur_amount)
    alpha_blurred = np.clip(alpha_blurred, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha_blurred
    return Image.fromarray(arr)

#Checks if the uploaded image has transparency. If not (e.g., a JPG), it guesses the background color and creates a mask.
def ensure_alpha(img: Image.Image) -> Tuple[Image.Image, str]:
    img = img.convert("RGBA")
    arr = np.array(img)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        if alpha.min() < 255:
            return img, "native_alpha"

    rgb = arr[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]
    pad = max(4, int(min(h, w) * 0.02))
    samples = np.concatenate(
        [
            rgb[:pad, :, :].reshape(-1, 3),
            rgb[-pad:, :, :].reshape(-1, 3),
            rgb[:, :pad, :].reshape(-1, 3),
            rgb[:, -pad:, :].reshape(-1, 3),
        ],
        axis=0,
    )
    bg_color = np.median(samples, axis=0)
    dist = np.sqrt(((rgb - bg_color) ** 2).sum(axis=2))
    med = np.median(dist)
    thr = float(max(20.0, med * 1.2))
    mask = (dist > thr).astype(np.uint8) * 255
    mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
    mask = np.clip(mask, 0, 255).astype(np.uint8)
    arr[:, :, 3] = mask
    return Image.fromarray(arr), "computed_mask"


def _letterbox_rgba(img_rgba: Image.Image, size: int, pad_rgba=(0, 0, 0, 0)) -> Image.Image:
    img = img_rgba.convert("RGBA")
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (size, size), pad_rgba)
    scale = min(size / w, size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

    canvas = Image.new("RGBA", (size, size), pad_rgba)
    off_x = (size - new_w) // 2
    off_y = (size - new_h) // 2
    canvas.alpha_composite(resized, dest=(off_x, off_y))
    return canvas


def _cover_rgba(img_rgba: Image.Image, size: int) -> Image.Image:
    img = img_rgba.convert("RGBA")
    w, h = img.size
    if w == 0 or h == 0:
        return Image.new("RGBA", (size, size), (0, 0, 0, 0))
    scale = max(size / w, size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    right = left + size
    bottom = top + size
    return resized.crop((left, top, right, bottom))


def standardize_canvas(
    img_rgba: Image.Image,
    size: int,
    mode: str,
    pad_rgba: Tuple[int, int, int, int],
) -> Image.Image:
    if size is None or size <= 0:
        return img_rgba.convert("RGBA")
    if "Letterbox" in mode:
        return _letterbox_rgba(img_rgba, size, pad_rgba)
    return _cover_rgba(img_rgba, size)


def _save_png_to_temp(image: Image.Image, prefix: str = "aivion") -> Optional[str]:
    if image is None:
        return None
    image = image.convert("RGBA")
    tmpdir = tempfile.gettempdir()
    ts = int(time.time() * 1000)
    path = os.path.join(tmpdir, f"{prefix}_{ts}.png")
    image.save(path, "PNG")
    return path


def _scale_rgba(img_rgba: Image.Image, scale_pct: int) -> Image.Image:
    scale = max(1, int(scale_pct)) / 100.0
    w, h = img_rgba.size
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return img_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _anchor_base_xy(bg_w: int, bg_h: int, fg_w: int, fg_h: int, anchor: str) -> Tuple[int, int]:
    anchors = {
        "Top-Left": (0, 0),
        "Top": ((bg_w - fg_w) // 2, 0),
        "Top-Right": (bg_w - fg_w, 0),
        "Left": (0, (bg_h - fg_h) // 2),
        "Center": ((bg_w - fg_w) // 2, (bg_h - fg_h) // 2),
        "Right": (bg_w - fg_w, (bg_h - fg_h) // 2),
        "Bottom-Left": (0, bg_h - fg_h),
        "Bottom": ((bg_w - fg_w) // 2, bg_h - fg_h),
        "Bottom-Right": (bg_w - fg_w, bg_h - fg_h),
    }
    return anchors.get(anchor, anchors["Center"])


# =========================
#  Core background removal
# =========================

#the primary pipeline function for Tab 1. It orchestrates the entire background removal process
def remove_bg_enhanced(
    input_img: Image.Image,
    model_choice: str,
    smoothing: float,
    apply_preprocess: bool,
    brightness: float,
    sharpness: float,
    saturation: float,
    auto_wb: bool,
    temperature: int,
    std_on: bool,
    std_size: int,
    std_mode: str,
    std_transparent: bool,
    std_pad_color: str,
):
    if input_img is None:
        return None, "No input image provided."

    img = input_img.convert("RGBA")
    if apply_preprocess:
        img = enhanced_preprocess_image(
            img,
            do_resize=True,
            brightness=brightness,
            sharpness=sharpness,
            saturation=saturation,
            auto_wb=auto_wb,
            temperature=int(temperature),
        )

    if "isnet-general-use" in model_choice:
        out = remove(img, session=session_isnet)
    elif "silueta" in model_choice:
        out = remove(img, session=session_silueta)
    else:
        out = remove(img, session=session_u2net)

    if smoothing and smoothing > 0:
        out = smooth_edges_only(out, smoothing)

    if std_on:
        pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
        out = standardize_canvas(out, int(std_size), std_mode, pad_rgba)

    out_checked, flag = ensure_alpha(out)
    arr = np.array(out_checked.convert("RGBA"))
    alpha = arr[:, :, 3]
    debug = (
        f"[ENHANCED] Mode: {out_checked.mode} | Size: {out_checked.size} | "
        f"Alpha min/max: {int(alpha.min())}/{int(alpha.max())} | "
        f"alpha_source: {flag} | standardized: {bool(std_on)}"
    )
    return out_checked, debug

#bg replacement logic
def replace_bg_step(
    removed_img: Image.Image,
    replace_option: str,
    color_choice: str,
    bg_upload: Optional[Image.Image],
    std_on: bool,
    std_size: int,
    std_mode: str,
    std_transparent: bool,
    std_pad_color: str,
    scale_pct: int = 100,
    offset_x: int = 0,
    offset_y: int = 0,
    anchor: str = "Center",
    clamp_inside: bool = True,
):
    if removed_img is None:
        return None, "No removed image available. Run Remove Background first."

    fg, flag = ensure_alpha(removed_img.convert("RGBA"))
    fg = _scale_rgba(fg, int(scale_pct))
    fg_w, fg_h = fg.size

    if std_on:
        canvas_w = canvas_h = int(std_size)
    else:
        canvas_w, canvas_h = removed_img.size

    # Background
    if replace_option is None or replace_option == "No Replacement":
        bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        debug = f"No replacement applied. alpha_source: {flag}"
    elif replace_option == "Solid Color":
        rgba = parse_color(color_choice)
        bg = Image.new("RGBA", (canvas_w, canvas_h), rgba)
        debug = f"Applied solid color {rgba} | alpha_source: {flag}"
    elif replace_option == "Upload Image":
        if bg_upload is None:
            bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            debug = "No background image uploaded; using transparent canvas."
        else:
            bg_src = bg_upload.convert("RGBA")
            side = min(canvas_w, canvas_h)
            if "Letterbox" in std_mode:
                fitted = _letterbox_rgba(bg_src, side, pad_rgba=(0, 0, 0, 0))
                bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                bg.alpha_composite(
                    fitted,
                    dest=((canvas_w - fitted.width) // 2, (canvas_h - fitted.height) // 2),
                )
            else:
                fitted = _cover_rgba(bg_src, side)
                bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                bg.alpha_composite(
                    fitted,
                    dest=((canvas_w - side) // 2, (canvas_h - side) // 2),
                )
            debug = f"Applied uploaded background (fitted to {canvas_w}x{canvas_h}) | alpha_source: {flag}"
    else:
        bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        debug = "Unknown option; using transparent canvas."

    base_x, base_y = _anchor_base_xy(canvas_w, canvas_h, fg_w, fg_h, anchor)
    x = int(base_x + offset_x)
    y = int(base_y + offset_y)

    if clamp_inside:
        x = max(0, min(x, canvas_w - fg_w))
        y = max(0, min(y, canvas_h - fg_h))

    composed = bg.copy()
    composed.paste(fg, (x, y), fg)

    debug += f" | placed at: ({x},{y}), scale: {scale_pct}%"
    if std_on:
        debug += f" | standardized canvas: {canvas_w}px {std_mode}"
    return composed, debug


def _load_image_from_any(file_obj: Any) -> Tuple[Optional[Image.Image], str]:
    """Helper to robustly load images from File component objects."""
    if isinstance(file_obj, str):
        name_show = os.path.basename(file_obj)
        if os.path.exists(file_obj):
            return Image.open(file_obj).convert("RGBA"), name_show
        return None, name_show

    if isinstance(file_obj, dict):
        data = file_obj.get("data", None)
        if data is not None:
            try:
                return (
                    Image.open(io.BytesIO(data)).convert("RGBA"),
                    file_obj.get("orig_name") or file_obj.get("name") or "uploaded",
                )
            except Exception:
                pass
        path = file_obj.get("path") or file_obj.get("name")
        if path and os.path.exists(path):
            return Image.open(path).convert("RGBA"), os.path.basename(path)
        return None, file_obj.get("orig_name") or file_obj.get("name") or "file"

    path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
    if path and isinstance(path, str) and os.path.exists(path):
        return Image.open(path).convert("RGBA"), os.path.basename(path)

    return None, "file"

#Processes multiple images in a loop and packages them into a ZIP file for easy download.
def remove_bg_batch(
    input_files,
    model_choice: str,
    smoothing: float,
    std_on: bool,
    std_size: int,
    std_mode: str,
    std_transparent: bool,
    std_pad_color: str,
):
    if not input_files:
        return [], None, "No files provided."

    input_files = input_files[:4]  # keep it small and friendly(max 4)
    results: List[Image.Image] = []
    msgs = []

    tmpdir = tempfile.gettempdir()
    zip_path = os.path.join(tmpdir, f"aivion_batch_{int(time.time())}.zip")
    zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    for idx, f in enumerate(input_files, start=1):
        img, display_name = _load_image_from_any(f)
        if img is None:
            msgs.append(f"[{idx}] Skipped: couldn't read {display_name}")
            continue

        try:
            if "isnet-general-use" in model_choice:
                out = remove(img, session=session_isnet)
            elif "silueta" in model_choice:
                out = remove(img, session=session_silueta)
            else:
                out = remove(img, session=session_u2net)

            if smoothing and smoothing > 0:
                out = smooth_edges_only(out, smoothing)

            if std_on:
                pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
                out = standardize_canvas(out, int(std_size), std_mode, pad_rgba)

            stem = os.path.splitext(os.path.basename(display_name))[0] or f"image_{idx}"
            out_path = os.path.join(tmpdir, f"{stem}_removed.png")
            out.save(out_path, "PNG")
            zf.write(out_path, arcname=os.path.basename(out_path))

            results.append(out)
            msgs.append(f"[{idx}] OK: {display_name} → removed")
        except Exception as e:
            msgs.append(f"[{idx}] Error on {display_name}: {e}")

    zf.close()
    zip_out = zip_path if len(results) > 0 else None
    status = "\n".join(msgs) if msgs else "Done."
    return results, zip_out, status


def enhance_only_preview(
    input_img: Image.Image,
    brightness: float,
    sharpness: float,
    saturation: float,
    auto_wb: bool,
    temperature: int,
    std_on: bool,
    std_size: int,
    std_mode: str,
    std_transparent: bool,
    std_pad_color: str,
):
    if input_img is None:
        return None, "No input image provided."

    img = enhanced_preprocess_image(
        input_img,
        do_resize=False,
        brightness=brightness,
        sharpness=sharpness,
        saturation=saturation,
        auto_wb=auto_wb,
        temperature=int(temperature),
    )

    if std_on:
        pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
        img = standardize_canvas(img, int(std_size), std_mode, pad_rgba)

    dbg = f"[PREVIEW] Size: {img.size} | std: {bool(std_on)}"
    return img, dbg

#Runs all three available AI models on the same image side-by-side so the user can pick the best one.
def test_all_models(image: Image.Image):
    if image is None:
        return None, None, None

    img = image.convert("RGBA")
    out_u2net = out_isnet = out_silueta = None

    try:
        out_u2net = remove(img, session=session_u2net)
    except Exception:
        out_u2net = None
    try:
        out_isnet = remove(img, session=session_isnet)
    except Exception:
        out_isnet = None
    try:
        out_silueta = remove(img, session=session_silueta)
    except Exception:
        out_silueta = None

    return out_u2net, out_isnet, out_silueta


# =========================
#  Trellis 3D / GLB
# =========================
GLB_OUTPUT_DIR = "outputs_glb"
os.makedirs(GLB_OUTPUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

trellis_pipe = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
trellis_pipe.to(device)


def generate_glb_trellis(
    mode: str,
    single_image: Image.Image,
    multi_images: Any,
):
    """
    Generate a GLB using Trellis.

    mode:
      - "Single image (one photo)"
      - "Multi-view (2–4 photos of same object)"
    """
    # -------- SINGLE IMAGE MODE --------
    if mode.startswith("Single"):
        if single_image is None:
            return None, None

        rgb = single_image.convert("RGB")

        outputs = trellis_pipe.run(
            rgb,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
        )
    # -------- MULTI-IMAGE MODE --------
    else:
        if not multi_images:
            return None, None

        pil_list: List[Image.Image] = []
        for f in multi_images[:4]:  # cap at 4 views
            img, _ = _load_image_from_any(f)
            if img is not None:
                pil_list.append(img.convert("RGB"))

        if len(pil_list) == 0:
            return None, None

        outputs = trellis_pipe.run_multi_image(
            pil_list,
            formats=["gaussian", "mesh"],
            preprocess_image=True,
        )

    # -------- COMMON EXPORT --------
    gs = outputs["gaussian"][0]
    mesh = outputs["mesh"][0]

    ts = int(time.time())
    slug = "multi" if mode.startswith("Multi") else "single"
    glb_name = f"aivion_{slug}_{ts}.glb"
    glb_path = os.path.join(GLB_OUTPUT_DIR, glb_name)

    glb = postprocessing_utils.to_glb(
        gs,
        mesh,
        simplify=0.95,
        texture_size=1024,
    )
    glb.export(glb_path)

    return glb_path, glb_path  # viewer path, download path



# =========================
#  UI Layout with Custom CSS
# =========================
custom_css = """
/* Import modern font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global styles */
* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}

/* Main container with gradient background */
.gradio-container {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%) !important;
    background-attachment: fixed !important;
}

/* Content wrapper with glassmorphism */
.contain {
    background: rgba(255, 255, 255, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 24px !important;
    padding: 2rem !important;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1) !important;
    margin: 1rem !important;
}

/* Header styling */
h1, .markdown h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700 !important;
    font-size: 3rem !important;
    margin-bottom: 0.5rem !important;
    text-align: center !important;
    letter-spacing: -0.02em !important;
}

/* Subtitle styling */
.markdown p {
    color: #64748b !important;
    font-size: 1.1rem !important;
    text-align: center !important;
    margin-bottom: 2rem !important;
    font-weight: 400 !important;
}

/* Tab styling */
.tab-nav {
    background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%) !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
    margin-bottom: 2rem !important;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05) !important;
}

.tab-nav button {
    border-radius: 12px !important;
    font-weight: 500 !important;
    transition: all 0.3s ease !important;
    color: #64748b !important;
    padding: 0.75rem 1.5rem !important;
    border: none !important;
}

.tab-nav button:hover {
    background: rgba(102, 126, 234, 0.1) !important;
    color: #667eea !important;
    transform: translateY(-2px) !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

/* Primary button styling */
.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(102, 126, 234, 0.3) !important;
    text-transform: none !important;
}

.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(102, 126, 234, 0.4) !important;
}

.primary:active {
    transform: translateY(0) !important;
}

/* Secondary button styling */
.secondary {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.875rem 2rem !important;
    font-size: 1rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 16px rgba(240, 147, 251, 0.3) !important;
}

.secondary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(240, 147, 251, 0.4) !important;
}

/* Input fields */
input, textarea, select {
    border: 2px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.75rem !important;
    transition: all 0.3s ease !important;
    background: white !important;
}

input:focus, textarea:focus, select:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    outline: none !important;
}

/* Labels */
label {
    font-weight: 500 !important;
    color: #334155 !important;
    margin-bottom: 0.5rem !important;
    font-size: 0.95rem !important;
}

/* Image containers */
.gr-image {
    border-radius: 16px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.08) !important;
    transition: all 0.3s ease !important;
    border: 2px solid #e2e8f0 !important;
}

.gr-image:hover {
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.12) !important;
    transform: translateY(-2px) !important;
}

/* Sliders */
input[type="range"] {
    accent-color: #667eea !important;
}

/* Checkboxes and radio buttons */
input[type="checkbox"], input[type="radio"] {
    accent-color: #667eea !important;
    width: 1.25rem !important;
    height: 1.25rem !important;
}

/* File upload area */
.file-preview {
    border: 2px dashed #cbd5e1 !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%) !important;
    transition: all 0.3s ease !important;
}

.file-preview:hover {
    border-color: #667eea !important;
    background: linear-gradient(135deg, #f8fafc 0%, #ede9fe 100%) !important;
}

/* Markdown sections */
.markdown h2, .markdown h3 {
    color: #1e293b !important;
    font-weight: 600 !important;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}

.markdown h3 {
    font-size: 1.25rem !important;
    color: #475569 !important;
}

/* Textbox */
.textbox {
    border-radius: 10px !important;
    border: 2px solid #e2e8f0 !important;
    background: #f8fafc !important;
}

/* Color picker */
input[type="color"] {
    border-radius: 8px !important;
    border: 2px solid #e2e8f0 !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
}

input[type="color"]:hover {
    border-color: #667eea !important;
    transform: scale(1.05) !important;
}

/* Scrollbar styling */
::-webkit-scrollbar {
    width: 10px !important;
}

::-webkit-scrollbar-track {
    background: #f1f5f9 !important;
    border-radius: 10px !important;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border-radius: 10px !important;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
}

/* Divider */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, #e2e8f0, transparent) !important;
    margin: 2rem 0 !important;
}
"""

with gr.Blocks(css=custom_css, theme=gr.themes.Soft()) as demo:
    gr.Markdown("# ✨ AIVION")
    gr.Markdown(
        "🎨 **Professional Background Removal** • 🖼️ **Image Enhancement** • 🔍 **Model Comparison** • 🎭 **3D Generation**"
    )

    # TAB 1: Background Removal + Replacement
    with gr.Tab("🎨 Background Removal & Replacement"):
        with gr.Row():
            with gr.Column(scale=1):
                in_img = gr.Image(type="pil", label="Upload Single Product Image")
                multi_imgs = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="📁 Or upload multiple images (2–4 recommended)",
                )

                model_sel = gr.Radio(
                    choices=[
                        "isnet-general-use (Recommended for Products)",
                        "u2net (Default)",
                        "silueta (Objects)",
                    ],
                    value="isnet-general-use (Recommended for Products)",
                    label="Choose Model",
                )
                smooth = gr.Slider(
                    minimum=0, maximum=3, value=0, step=0.5, label="Edge Smoothing"
                )

                gr.Markdown("### Output Size")
                std_on = gr.Checkbox(label="Standardize output size (square)", value=True)
                with gr.Row():
                    std_size = gr.Dropdown(
                        choices=[512, 768, 1024],
                        value=1024,
                        label="Target size (px)",
                    )
                    std_mode = gr.Radio(
                        choices=["Letterbox (pad)", "Cover (crop)"],
                        value="Letterbox (pad)",
                        label="Fill mode",
                    )
                with gr.Row():
                    std_transparent = gr.Checkbox(
                        label="Pad with transparency (RGBA)", value=True
                    )
                    std_pad_color = gr.ColorPicker(
                        label="Pad color (when transparency is OFF)", value="#ffffff"
                    )

                btn_enh = gr.Button(
                    "🗡 Remove Background (Single)", variant="primary"
                )

                gr.Markdown("---")
                gr.Markdown("### Batch Remove Backgrounds")
                btn_batch = gr.Button(
                    "✨ Remove Backgrounds (Batch)", variant="secondary"
                )

            with gr.Column(scale=1):
                gr.Markdown("### Removed (transparent) — Single")
                out_enh = gr.Image(
                    type="pil", label="Removed Image", elem_classes="gr-image"
                )
                dbg_enh = gr.Textbox(label="Debug", interactive=False)

                dl_removed_btn = gr.Button("⬇️ Download Removed PNG")
                dl_removed_file = gr.File(label="Download: removed.png")

                gr.Markdown("### Replacement (Single)")
                rep_sel = gr.Radio(
                    choices=["No Replacement", "Solid Color", "Upload Image"],
                    value="No Replacement",
                    label="Replace with",
                )
                rep_color = gr.ColorPicker(
                    label="Pick solid color (if chosen)", value="#ffffff"
                )
                rep_bg = gr.Image(
                    type="pil", label="Upload background image (if chosen)"
                )

                gr.Markdown("#### Positioning")
                scale_pct_ui = gr.Slider(
                    minimum=30,
                    maximum=200,
                    value=100,
                    step=1,
                    label="Product Scale (%)",
                )
                anchor_ui = gr.Dropdown(
                    choices=[
                        "Top-Left",
                        "Top",
                        "Top-Right",
                        "Left",
                        "Center",
                        "Right",
                        "Bottom-Left",
                        "Bottom",
                        "Bottom-Right",
                    ],
                    value="Center",
                    label="Anchor",
                )
                offset_x_ui = gr.Slider(
                    minimum=-1200,
                    maximum=1200,
                    value=0,
                    step=1,
                    label="Offset X (px)",
                )
                offset_y_ui = gr.Slider(
                    minimum=-1200,
                    maximum=1200,
                    value=0,
                    step=1,
                    label="Offset Y (px)",
                )
                clamp_ui = gr.Checkbox(
                    label="Keep product fully inside canvas", value=True
                )

                btn_replace = gr.Button("🖌 Apply Replacement", variant="secondary")

                gr.Markdown("### Final Result (Single)")
                out_final = gr.Image(
                    type="pil", label="Final Result", elem_classes="gr-image"
                )
                dbg_final = gr.Textbox(
                    label="Replacement debug info", interactive=False
                )

                dl_final_btn = gr.Button("⬇️ Download Final PNG")
                dl_final_file = gr.File(label="Download: final.png")

                gr.Markdown("---")
                gr.Markdown("### Batch Results (Preview)")
                batch_preview = gr.Gallery(
                    label="Batch Preview", columns=2, height="auto"
                )
                batch_zip = gr.File(label="⬇️ Download All (ZIP)")
                batch_status = gr.Textbox(label="Batch Status", interactive=False)

        btn_enh.click(
            fn=remove_bg_enhanced,
            inputs=[
                in_img,
                model_sel,
                smooth,
                gr.State(False),  # apply_preprocess = False
                gr.State(1.0),
                gr.State(1.0),
                gr.State(1.0),
                gr.State(False),
                gr.State(0),
                std_on,
                std_size,
                std_mode,
                std_transparent,
                std_pad_color,
            ],
            outputs=[out_enh, dbg_enh],
        )

        btn_replace.click(
            fn=replace_bg_step,
            inputs=[
                out_enh,
                rep_sel,
                rep_color,
                rep_bg,
                std_on,
                std_size,
                std_mode,
                std_transparent,
                std_pad_color,
                scale_pct_ui,
                offset_x_ui,
                offset_y_ui,
                anchor_ui,
                clamp_ui,
            ],
            outputs=[out_final, dbg_final],
        )

        dl_removed_btn.click(
            fn=_save_png_to_temp, inputs=[out_enh], outputs=[dl_removed_file]
        )
        dl_final_btn.click(
            fn=_save_png_to_temp, inputs=[out_final], outputs=[dl_final_file]
        )

        btn_batch.click(
            fn=remove_bg_batch,
            inputs=[
                multi_imgs,
                model_sel,
                smooth,
                std_on,
                std_size,
                std_mode,
                std_transparent,
                std_pad_color,
            ],
            outputs=[batch_preview, batch_zip, batch_status],
        )

    # TAB 2: Enhancements
    with gr.Tab("🖼️ Enhancements (Preview Only)"):
        with gr.Row():
            with gr.Column(scale=1):
                in_prev = gr.Image(type="pil", label="Upload Product Image")

                gr.Markdown("### Adjustments")
                brightness_p = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.05, label="Brightness"
                )
                sharpness_p = gr.Slider(
                    0.5, 2.0, value=1.0, step=0.05, label="Sharpness / Clarity"
                )
                saturation_p = gr.Slider(
                    0.5, 1.5, value=1.0, step=0.05, label="Color (Saturation)"
                )
                auto_wb_p = gr.Checkbox(
                    label="Auto White Balance (Gray-World)", value=False
                )
                temp_p = gr.Slider(
                    -50, 50, value=0, step=1, label="Temperature (Cool ↔ Warm)"
                )

                gr.Markdown("### Output Size (optional)")
                std_on_p = gr.Checkbox(
                    label="Standardize output size (square)", value=False
                )
                with gr.Row():
                    std_size_p = gr.Dropdown(
                        choices=[512, 768, 1024],
                        value=1024,
                        label="Target size (px)",
                    )
                    std_mode_p = gr.Radio(
                        choices=["Letterbox (pad)", "Cover (crop)"],
                        value="Letterbox (pad)",
                        label="Fill mode",
                    )
                with gr.Row():
                    std_transparent_p = gr.Checkbox(
                        label="Pad with transparency (RGBA)", value=True
                    )
                    std_pad_color_p = gr.ColorPicker(
                        label="Pad color (when transparency is OFF)", value="#ffffff"
                    )

                btn_preview = gr.Button("✨ Preview Enhancements", variant="secondary")
                dl_preview_btn = gr.Button("⬇️ Download Preview PNG")
                dl_preview_file = gr.File(label="Download: preview.png")

            with gr.Column(scale=1):
                prev_img = gr.Image(
                    type="pil", label="Enhanced Preview", elem_classes="gr-image"
                )
                prev_dbg = gr.Textbox(label="Preview debug", interactive=False)

        btn_preview.click(
            fn=enhance_only_preview,
            inputs=[
                in_prev,
                brightness_p,
                sharpness_p,
                saturation_p,
                auto_wb_p,
                temp_p,
                std_on_p,
                std_size_p,
                std_mode_p,
                std_transparent_p,
                std_pad_color_p,
            ],
            outputs=[prev_img, prev_dbg],
        )
        dl_preview_btn.click(
            fn=_save_png_to_temp, inputs=[prev_img], outputs=[dl_preview_file]
        )

    # TAB 3: Model Comparison
    with gr.Tab("🔍 Model Comparison"):
        gr.Markdown("Compare the three models side-by-side")
        with gr.Row():
            comp_input = gr.Image(type="pil", label="Upload Product Image")
            compare_btn = gr.Button("🚀 Run Comparison")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### u2net (Default)")
                cmp_u2 = gr.Image(type="pil", elem_classes="gr-image")
            with gr.Column():
                gr.Markdown("### isnet-general-use")
                cmp_is = gr.Image(type="pil", elem_classes="gr-image")
            with gr.Column():
                gr.Markdown("### silueta")
                cmp_si = gr.Image(type="pil", elem_classes="gr-image")

        compare_btn.click(
            fn=test_all_models, inputs=[comp_input], outputs=[cmp_u2, cmp_is, cmp_si]
        )

    # TAB 4: Trellis 3D GLB
    with gr.Tab("🎭 3D GLB Generator (Trellis)"):
        gr.Markdown(
            "Upload cleaned product images (after background removal) to generate a 3D GLB.\n\n"
            "- **Single image**: one good front-facing shot\n"
            "- **Multi-view**: 2–4 images of the SAME object from different angles"
        )

        with gr.Row():
            with gr.Column(scale=1):
                trellis_mode = gr.Radio(
                    choices=[
                        "Single image (one photo)",
                        "Multi-view (2–4 photos of same object)",
                    ],
                    value="Single image (one photo)",
                    label="Mode",
                )

                trellis_input_single = gr.Image(
                    type="pil",
                    label="Single product image (PNG/JPG)",
                )

                trellis_input_multi = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Multi-view images (2–4). Use different angles.",
                )

                gen_glb_btn = gr.Button("🎨 Generate GLB", variant="primary")

            with gr.Column(scale=1):
                glb_viewer = gr.Model3D(label="3D Preview")
                glb_file = gr.File(label="Download GLB")

        gen_glb_btn.click(
            fn=generate_glb_trellis,
            inputs=[trellis_mode, trellis_input_single, trellis_input_multi],
            outputs=[glb_viewer, glb_file],
        )

    # TAB 5: About
    with gr.Tab("ℹ️ About"):
        gr.Markdown(
            """
## Best Practices for Product Images
- Keep your camera steady
- Ensure **good lighting**
- Avoid heavy shadows and cluttered backgrounds
- Center your product in the frame

### Model Guide
- **isnet-general-use** → Best for most e-commerce products
- **u2net** → General-purpose, works on many scenes
- **silueta** → Simple objects / clear silhouettes

### Workflow
1. Tab 1 → Remove background (single or batch), apply replacement, download final PNGs
2. Tab 2 → Fine-tune brightness / color / sharpness and export previews
3. Tab 3 → Compare model outputs quickly
4. Tab 4 → Take your cleaned PNG or multi-view shots → generate 3D GLB + view it in the browser
"""
        )

    gr.HTML(
        "<div style='text-align:center; padding:12px; color:#999;'>© 2025 Aivion</div>"
    )


if __name__ == "__main__":
    demo.launch(share=True)
