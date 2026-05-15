import os
import io
import time
import tempfile
import zipfile
import subprocess
import shutil
from typing import Any, Optional, Tuple, List
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from rembg import remove, new_session
from scipy.ndimage import gaussian_filter
import gradio as gr
from groq import Groq
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Cleanup old temp files
subprocess.run("rm -rf /tmp/tmp*", shell=True)

# --- PATHS ---
VIDEO_ENV_PYTHON = "/home/ubuntu/aivion-video/bin/python"
VIDEO_SCRIPT_PATH = "/home/ubuntu/video2.py"
TRELLIS_ENV_PYTHON = "/home/ubuntu/TRELLIS/trellis-venv/bin/python"
TRELLIS_WORKER = "/home/ubuntu/TRELLIS/trellis_worker.py"
TRELLIS_DIR = "/home/ubuntu/TRELLIS"

# --- CONFIG & CLIENTS ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "MY_GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
VISION_MODEL = "gemini-2.0-flash-lite"

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "MY_GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
TEXT_MODEL = "llama-3.3-70b-versatile"

TOP_K = 5
DATASET_PATH = "Social Media Engagement Dataset.csv"

# Initialize rembg sessions (CPU mode for stability)
sess_isnet = new_session("isnet-general-use", providers=['CPUExecutionProvider'])
sess_u2net = new_session("u2net", providers=['CPUExecutionProvider'])
sess_silueta = new_session("silueta", providers=['CPUExecutionProvider'])

# =========================
# DATA LOADING & RAG SETUP
# =========================
def load_dataset(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()
        if "text_content" in df.columns:
            df = df.rename(columns={"text_content": "caption"})
        if "caption" not in df.columns: raise ValueError("No caption found.")
        if "hashtags" not in df.columns: df["hashtags"] = ""
        df = df.dropna(subset=["caption"]).copy()
        return df
    except Exception as e:
        return pd.DataFrame([{"caption": "Sample post", "hashtags": "#sample", "platform": "instagram"}])

df_rag = load_dataset(DATASET_PATH)

class EngagementRAG:
    def __init__(self, dataframe: pd.DataFrame, top_k: int = 5):
        self.df = dataframe.copy()
        self.top_k = top_k
        self._build_index()

    def _build_index(self):
        self.df["search_text"] = self.df["caption"].fillna("") + " " + self.df["hashtags"].fillna("")
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df["search_text"].tolist())

    def retrieve(self, query: str, platform: str = "all") -> pd.DataFrame:
        pool = self.df.copy()
        if platform != "all" and "platform" in pool.columns:
            filtered = pool[pool["platform"].str.lower() == platform.lower()]
            if len(filtered) >= self.top_k: pool = filtered
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.tfidf_matrix[pool.index]).flatten()
        top_indices = np.argsort(sims)[::-1][:self.top_k]
        return pool.iloc[top_indices]

    def format_examples(self, retrieved: pd.DataFrame) -> str:
        lines = [f"Example {i+1}:\n  Caption: {row['caption']}\n  Hashtags: {row['hashtags']}" for i, (_, row) in enumerate(retrieved.iterrows())]
        return "\n\n".join(lines)

rag = EngagementRAG(df_rag, top_k=TOP_K)

# =========================
# BG REPLACEMENT UTILITIES (From Trellis UI)
# =========================
def parse_color(color_str: str) -> Tuple[int, int, int, int]:
    if not color_str:
        return (255, 255, 255, 255)
    s = color_str.strip()

    if s.startswith("#"):
        s = s.lstrip("#")
        if len(s) == 6:
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
            return (r, g, b, 255)
        if len(s) == 8:
            r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16); a = int(s[6:8], 16)
            return (r, g, b, a)

    if s.lower().startswith("rgb"):
        inside = s[s.find("(") + 1 : s.rfind(")")]
        parts = [p.strip() for p in inside.split(",")]
        try:
            r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
            if len(parts) == 4:
                a = float(parts[3])
                a = int(a * 255) if a <= 1 else int(a)
            else:
                a = 255
            return (r, g, b, a)
        except Exception:
            return (255, 255, 255, 255)

    return (255, 255, 255, 255)

def _auto_white_balance_gray_world(img_rgb: np.ndarray) -> np.ndarray:
    eps = 1e-6
    r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
    mean_r = r.mean() + eps; mean_g = g.mean() + eps; mean_b = b.mean() + eps
    mean_gray = (mean_r + mean_g + mean_b) / 3.0
    scale_r = mean_gray / mean_r; scale_g = mean_gray / mean_g; scale_b = mean_gray / mean_b

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
    image: Image.Image, do_resize: bool = True, brightness: float = 1.0,
    sharpness: float = 1.0, saturation: float = 1.0, auto_wb: bool = False, temperature: int = 0,
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

    if saturation != 1.0: rgb = ImageEnhance.Color(rgb).enhance(float(saturation))
    if brightness != 1.0: rgb = ImageEnhance.Brightness(rgb).enhance(float(brightness))
    if sharpness != 1.0: rgb = ImageEnhance.Sharpness(rgb).enhance(float(sharpness))

    img = rgb.convert("RGBA")
    img = ImageEnhance.Contrast(img).enhance(1.15)

    if do_resize and max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    return img

def smooth_edges_only(image: Image.Image, blur_amount: float = 1.0) -> Image.Image:
    if blur_amount == 0: return image
    img = image.convert("RGBA")
    arr = np.array(img)
    if arr.shape[2] < 4: return image
    alpha = arr[:, :, 3].astype(np.float32)
    alpha_blurred = gaussian_filter(alpha, sigma=blur_amount)
    alpha_blurred = np.clip(alpha_blurred, 0, 255).astype(np.uint8)
    arr[:, :, 3] = alpha_blurred
    return Image.fromarray(arr)

def ensure_alpha(img: Image.Image) -> Tuple[Image.Image, str]:
    img = img.convert("RGBA")
    arr = np.array(img)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3]
        if alpha.min() < 255: return img, "native_alpha"

    rgb = arr[:, :, :3].astype(np.float32)
    h, w = rgb.shape[:2]
    pad = max(4, int(min(h, w) * 0.02))
    samples = np.concatenate(
        [rgb[:pad, :, :].reshape(-1, 3), rgb[-pad:, :, :].reshape(-1, 3),
         rgb[:, :pad, :].reshape(-1, 3), rgb[:, -pad:, :].reshape(-1, 3)], axis=0,
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

def _letterbox_rgba(img_rgba: Image.Image, size_w: int, size_h: int, pad_rgba=(0, 0, 0, 0)) -> Image.Image:
    img = img_rgba.convert("RGBA")
    w, h = img.size
    if w == 0 or h == 0: return Image.new("RGBA", (size_w, size_h), pad_rgba)
    scale = min(size_w / w, size_h / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size_w, size_h), pad_rgba)
    off_x = (size_w - new_w) // 2
    off_y = (size_h - new_h) // 2
    canvas.alpha_composite(resized, dest=(off_x, off_y))
    return canvas

def _cover_rgba(img_rgba: Image.Image, size: int) -> Image.Image:
    img = img_rgba.convert("RGBA")
    w, h = img.size
    if w == 0 or h == 0: return Image.new("RGBA", (size, size), (0, 0, 0, 0))
    scale = max(size / w, size / h)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
    left = (new_w - size) // 2
    top = (new_h - size) // 2
    return resized.crop((left, top, left + size, top + size))

def standardize_canvas(img_rgba: Image.Image, size: int, mode: str, pad_rgba: Tuple[int, int, int, int]) -> Image.Image:
    if size is None or size <= 0: return img_rgba.convert("RGBA")
    if "Letterbox" in mode: return _letterbox_rgba(img_rgba, size, size, pad_rgba)
    return _cover_rgba(img_rgba, size)

def _save_png_to_temp(image: Image.Image, prefix: str = "aivion") -> Optional[str]:
    if image is None: return None
    image = image.convert("RGBA")
    tmpdir = tempfile.gettempdir()
    ts = int(time.time() * 1000)
    path = os.path.join(tmpdir, f"{prefix}_{ts}.png")
    image.save(path, "PNG")
    return path

def _scale_rgba(img_rgba: Image.Image, scale_pct: int) -> Image.Image:
    scale = max(1, int(scale_pct)) / 100.0
    w, h = img_rgba.size
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    return img_rgba.resize((new_w, new_h), Image.Resampling.LANCZOS)

def _anchor_base_xy(bg_w: int, bg_h: int, fg_w: int, fg_h: int, anchor: str) -> Tuple[int, int]:
    anchors = {
        "Top-Left": (0, 0), "Top": ((bg_w - fg_w) // 2, 0), "Top-Right": (bg_w - fg_w, 0),
        "Left": (0, (bg_h - fg_h) // 2), "Center": ((bg_w - fg_w) // 2, (bg_h - fg_h) // 2),
        "Right": (bg_w - fg_w, (bg_h - fg_h) // 2), "Bottom-Left": (0, bg_h - fg_h),
        "Bottom": ((bg_w - fg_w) // 2, bg_h - fg_h), "Bottom-Right": (bg_w - fg_w, bg_h - fg_h),
    }
    return anchors.get(anchor, anchors["Center"])

def _load_image_from_any(file_obj: Any) -> Tuple[Optional[Image.Image], str]:
    if isinstance(file_obj, str):
        name_show = os.path.basename(file_obj)
        if os.path.exists(file_obj): return Image.open(file_obj).convert("RGBA"), name_show
        return None, name_show
    if isinstance(file_obj, dict):
        data = file_obj.get("data", None)
        if data is not None:
            try: return (Image.open(io.BytesIO(data)).convert("RGBA"), file_obj.get("orig_name") or file_obj.get("name") or "uploaded")
            except Exception: pass
        path = file_obj.get("path") or file_obj.get("name")
        if path and os.path.exists(path): return Image.open(path).convert("RGBA"), os.path.basename(path)
        return None, file_obj.get("orig_name") or file_obj.get("name") or "file"
    path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
    if path and isinstance(path, str) and os.path.exists(path): return Image.open(path).convert("RGBA"), os.path.basename(path)
    return None, "file"


# =========================
# CORE LOGIC FUNCTIONS
# =========================
def remove_bg_enhanced(
    input_img: Image.Image, model_choice: str, smoothing: float, apply_preprocess: bool,
    brightness: float, sharpness: float, saturation: float, auto_wb: bool, temperature: int,
    std_on: bool, std_size: int, std_mode: str, std_transparent: bool, std_pad_color: str,
):
    if input_img is None: return None, "No input image provided."
    img = input_img.convert("RGBA")
    if apply_preprocess:
        img = enhanced_preprocess_image(img, do_resize=True, brightness=brightness, sharpness=sharpness, saturation=saturation, auto_wb=auto_wb, temperature=int(temperature))
    if "isnet" in model_choice: out = remove(img, session=sess_isnet)
    elif "silueta" in model_choice: out = remove(img, session=sess_silueta)
    else: out = remove(img, session=sess_u2net)

    if smoothing and smoothing > 0: out = smooth_edges_only(out, smoothing)

    if std_on:
        pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
        out = standardize_canvas(out, int(std_size), std_mode, pad_rgba)

    out_checked, flag = ensure_alpha(out)
    arr = np.array(out_checked.convert("RGBA"))
    alpha = arr[:, :, 3]
    debug = f"[ENHANCED] Mode: {out_checked.mode} | Size: {out_checked.size} | Alpha min/max: {int(alpha.min())}/{int(alpha.max())} | alpha_source: {flag} | standardized: {bool(std_on)}"
    return out_checked, debug

def replace_bg_step(
    removed_img: Image.Image, replace_option: str, color_choice: str, bg_upload: Optional[Image.Image],
    std_on: bool, std_size: int, std_mode: str, std_transparent: bool, std_pad_color: str,
    scale_pct: int = 100, offset_x: int = 0, offset_y: int = 0, anchor: str = "Center", clamp_inside: bool = True,
):
    if removed_img is None: return None, "No removed image available. Run Remove Background first."
    fg, flag = ensure_alpha(removed_img.convert("RGBA"))
    fg = _scale_rgba(fg, int(scale_pct))
    fg_w, fg_h = fg.size

    if std_on: canvas_w = canvas_h = int(std_size)
    else: canvas_w, canvas_h = removed_img.size

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
                fitted = _letterbox_rgba(bg_src, side, side, pad_rgba=(0, 0, 0, 0))
                bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                bg.alpha_composite(fitted, dest=((canvas_w - fitted.width) // 2, (canvas_h - fitted.height) // 2))
            else:
                fitted = _cover_rgba(bg_src, side)
                bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
                bg.alpha_composite(fitted, dest=((canvas_w - side) // 2, (canvas_h - side) // 2))
            debug = f"Applied uploaded background (fitted to {canvas_w}x{canvas_h}) | alpha_source: {flag}"
    else:
        bg = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        debug = "Unknown option; using transparent canvas."

    base_x, base_y = _anchor_base_xy(canvas_w, canvas_h, fg_w, fg_h, anchor)
    x = int(base_x + offset_x); y = int(base_y + offset_y)
    if clamp_inside:
        x = max(0, min(x, canvas_w - fg_w))
        y = max(0, min(y, canvas_h - fg_h))

    composed = bg.copy()
    composed.paste(fg, (x, y), fg)
    debug += f" | placed at: ({x},{y}), scale: {scale_pct}%"
    if std_on: debug += f" | standardized canvas: {canvas_w}px {std_mode}"
    return composed, debug

def remove_bg_batch(
    input_files, model_choice: str, smoothing: float, std_on: bool, std_size: int,
    std_mode: str, std_transparent: bool, std_pad_color: str,
):
    if not input_files: return [], None, "No files provided."
    input_files = input_files[:4]
    results: List[Any] = []; msgs = []
    tmpdir = tempfile.gettempdir(); zip_path = os.path.join(tmpdir, f"aivion_batch_{int(time.time())}.zip")
    zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

    for idx, f in enumerate(input_files, start=1):
        img, display_name = _load_image_from_any(f)
        if img is None:
            msgs.append(f"[{idx}] Skipped: couldn't read {display_name}")
            continue
        try:
            if "isnet" in model_choice: out = remove(img, session=sess_isnet)
            elif "silueta" in model_choice: out = remove(img, session=sess_silueta)
            else: out = remove(img, session=sess_u2net)

            if smoothing and smoothing > 0: out = smooth_edges_only(out, smoothing)
            if std_on:
                pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
                out = standardize_canvas(out, int(std_size), std_mode, pad_rgba)

            stem = os.path.splitext(os.path.basename(display_name))[0] or f"image_{idx}"
            out_path = os.path.join(tmpdir, f"{stem}_removed.png"); out.save(out_path, "PNG")
            zf.write(out_path, arcname=os.path.basename(out_path))
            results.append(out_path); msgs.append(f"[{idx}] OK: {display_name} → removed")
        except Exception as e:
            msgs.append(f"[{idx}] Error on {display_name}: {e}")

    zf.close(); zip_out = zip_path if len(results) > 0 else None
    status = "\n".join(msgs) if msgs else "Done."
    return results, zip_out, status

def update_position_on_click(evt: gr.SelectData, removed_img, scale_pct, anchor, std_on, std_size):
    if removed_img is None: return 0, 0
    x_click, y_click = evt.index
    fg, _ = ensure_alpha(removed_img.convert("RGBA"))
    fg = _scale_rgba(fg, int(scale_pct))
    fg_w, fg_h = fg.size
    if std_on: canvas_w = canvas_h = int(std_size)
    else: canvas_w, canvas_h = removed_img.size
    base_x, base_y = _anchor_base_xy(canvas_w, canvas_h, fg_w, fg_h, anchor)
    target_x = x_click - fg_w // 2
    target_y = y_click - fg_h // 2
    return int(target_x - base_x), int(target_y - base_y)

def run_video_engine(img, template):
    if img is None: 
        yield gr.update(), "Please upload an image"
        return

    # CogVideoX-5B-I2V default resolution is 720x480. We letterbox pad the image to fit this natively without stretching.
    padded_img = _letterbox_rgba(img.convert("RGBA"), 720, 480, pad_rgba=(0,0,0,255)).convert("RGB")
    
    with tempfile.TemporaryDirectory() as tmp:
        t_in, t_out = os.path.join(tmp, "v.png"), os.path.join(tmp, "v.mp4")
        padded_img.save(t_in)
        proc = None
        try:
            start_t = time.time()
            proc = subprocess.Popen([VIDEO_ENV_PYTHON, VIDEO_SCRIPT_PATH, "--image", t_in, "--template", template, "--output", t_out])
            while proc.poll() is None:
                time.sleep(1)
                elapsed = int(time.time() - start_t)
                m, s = divmod(elapsed, 60)
                yield gr.update(), f"Processing video... (Elapsed: {m:02d}:{s:02d})"
            
            if proc.returncode == 0:
                stream_path = "aivion_video_result.mp4"
                shutil.copy(t_out, stream_path)
                yield stream_path, "Done!"
            else:
                yield gr.update(), "Error during rendering."
        finally:
            if proc and proc.poll() is None:
                proc.terminate()
                yield gr.update(), "Cancelled"

def run_3d_generation(img, multi_files, mode):
    out_p = os.path.abspath("aivion_output.glb")
    proc = None
    try:
        if "Multi" in mode and multi_files:
            cmd = [TRELLIS_ENV_PYTHON, TRELLIS_WORKER, "--images"] + [f.name for f in multi_files] + ["--output", out_p]
        else:
            if img is None: 
                yield gr.update()
                return
            img.save("t_temp.png")
            cmd = [TRELLIS_ENV_PYTHON, TRELLIS_WORKER, "--image", "t_temp.png", "--output", out_p]
            
        start_t = time.time()
        proc = subprocess.Popen(cmd, cwd=TRELLIS_DIR)
        while proc.poll() is None:
            time.sleep(1)
            # 3D Model viewer doesn't have a status text box attached to its output, 
            # so we just yield gr.update() to keep it alive
            yield gr.update()
            
        if proc.returncode == 0:
            yield out_p
        else:
            yield gr.update()
    finally:
        if proc and proc.poll() is None:
            proc.terminate()
            yield gr.update()

# --- CONTENT ENGINE LOGIC ---
context_store = {"last_output": "", "product_desc": ""}
TONE_MAP = {"Professional": "polished", "Casual & Fun": "relaxed", "Inspiring": "uplifting", "Urgent / FOMO": "exciting"}

def describe_image(image: Image.Image) -> str:
    if image is None: return ""
    buf = io.BytesIO(); image.save(buf, format="JPEG")
    try:
        res = gemini_client.models.generate_content(model=VISION_MODEL, contents=[types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg"), "Describe this product."])
        return res.text.strip()
    except: return "Product Image"

def run_generator(image, user_prompt, platform, tone, num_captions):
    yield "Analysing...", ""
    product_desc = describe_image(image) if image is not None else ""
    query = f"{product_desc} {user_prompt}".strip()
    retrieved = rag.retrieve(query, platform.lower() if platform != "All" else "all")
    examples_text = rag.format_examples(retrieved)
    system = (
        f"You are a social media expert for {platform}. Tone: {TONE_MAP.get(tone, tone)}.\n"
        "CRITICAL RULE: You MUST write captions ONLY for the product described in the Product section. "
        "If the User Instructions ask for a completely different product or contradict the image description, "
        "you must politely refuse and clarify that you can only write captions for the uploaded image."
    )
    user_section = f"Product: {product_desc}\nInstructions: {user_prompt}\nExamples:\n{examples_text}\nWrite {num_captions} variants."
    response = groq_client.chat.completions.create(model=TEXT_MODEL, messages=[{"role": "system", "content": system}, {"role": "user", "content": user_section}])
    context_store["last_output"] = response.choices[0].message.content.strip()
    yield "Done!", f"## Generated Content\n\n{context_store['last_output']}"


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

# =========================
# UI LAYOUT
# =========================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(), title="Aivion Multimodal OS") as demo:
    gr.Markdown("# Aivion Multimodal OS")

    with gr.Tabs():
        with gr.Tab("BG Removal & Batch"):
            with gr.Row():
                with gr.Column(scale=1):
                    in_img = gr.Image(type="pil", label="Upload Single Product Image")
                    multi_imgs = gr.File(file_count="multiple", file_types=["image"], label="Or upload multiple images (2-4 recommended)")

                    model_sel = gr.Radio(choices=["isnet-general-use (Recommended for Products)", "u2net (Default)", "silueta (Objects)"], value="isnet-general-use (Recommended for Products)", label="Choose Model")
                    smooth = gr.Slider(minimum=0, maximum=3, value=0.5, step=0.5, label="Edge Smoothing")

                    gr.Markdown("### Output Size")
                    std_on = gr.Checkbox(label="Standardize output size (square)", value=True)
                    with gr.Row():
                        std_size = gr.Dropdown(choices=[512, 768, 1024], value=1024, label="Target size (px)")
                        std_mode = gr.Radio(choices=["Letterbox (pad)", "Cover (crop)"], value="Letterbox (pad)", label="Fill mode")
                    with gr.Row():
                        std_transparent = gr.Checkbox(label="Pad with transparency (RGBA)", value=True)
                        std_pad_color = gr.ColorPicker(label="Pad color (when transparency is OFF)", value="#ffffff")

                    btn_enh = gr.Button("Remove Background (Single)", variant="primary")

                    gr.Markdown("---")
                    gr.Markdown("### Batch Remove Backgrounds")
                    btn_batch = gr.Button("Remove Backgrounds (Batch)", variant="secondary")

                with gr.Column(scale=1):
                    gr.Markdown("### Removed (transparent) - Single")
                    out_enh = gr.Image(type="pil", label="Removed Image", elem_classes="gr-image")
                    dbg_enh = gr.Textbox(label="Debug", interactive=False)

                    dl_removed_btn = gr.Button("Download Removed PNG")
                    dl_removed_file = gr.File(label="Download: removed.png")

                    gr.Markdown("### Replacement (Single)")
                    rep_sel = gr.Radio(choices=["No Replacement", "Solid Color", "Upload Image"], value="No Replacement", label="Replace with")
                    rep_color = gr.ColorPicker(label="Pick solid color (if chosen)", value="#ffffff")
                    rep_bg = gr.Image(type="pil", label="Upload background image (if chosen)")

                    gr.Markdown("#### Positioning")
                    scale_pct_ui = gr.Slider(minimum=30, maximum=200, value=100, step=1, label="Product Scale (%)")
                    anchor_ui = gr.Dropdown(choices=["Top-Left", "Top", "Top-Right", "Left", "Center", "Right", "Bottom-Left", "Bottom", "Bottom-Right"], value="Center", label="Anchor")
                    offset_x_ui = gr.Slider(minimum=-1200, maximum=1200, value=0, step=1, label="Offset X (px)")
                    offset_y_ui = gr.Slider(minimum=-1200, maximum=1200, value=0, step=1, label="Offset Y (px)")
                    clamp_ui = gr.Checkbox(label="Keep product fully inside canvas", value=True)

                    btn_replace = gr.Button("Apply Replacement", variant="secondary")

                    gr.Markdown("### Final Result (Single)")
                    out_final = gr.Image(type="pil", label="Final Result", elem_classes="gr-image")
                    dbg_final = gr.Textbox(label="Replacement debug info", interactive=False)

                    dl_final_btn = gr.Button("Download Final PNG")
                    dl_final_file = gr.File(label="Download: final.png")

                    gr.Markdown("---")
                    gr.Markdown("### Batch Results (Preview)")
                    batch_preview = gr.Gallery(label="Batch Preview", columns=2, height="auto")
                    batch_zip = gr.File(label="Download All (ZIP)")
                    batch_status = gr.Textbox(label="Batch Status", interactive=False)

            btn_enh.click(
                fn=remove_bg_enhanced,
                inputs=[in_img, model_sel, smooth, gr.State(False), gr.State(1.0), gr.State(1.0), gr.State(1.0), gr.State(False), gr.State(0), std_on, std_size, std_mode, std_transparent, std_pad_color],
                outputs=[out_enh, dbg_enh],
            )
            btn_replace.click(
                fn=replace_bg_step,
                inputs=[out_enh, rep_sel, rep_color, rep_bg, std_on, std_size, std_mode, std_transparent, std_pad_color, scale_pct_ui, offset_x_ui, offset_y_ui, anchor_ui, clamp_ui],
                outputs=[out_final, dbg_final],
            )
            out_final.select(fn=update_position_on_click, inputs=[out_enh, scale_pct_ui, anchor_ui, std_on, std_size], outputs=[offset_x_ui, offset_y_ui])
            
            dl_removed_btn.click(fn=_save_png_to_temp, inputs=[out_enh], outputs=[dl_removed_file])
            dl_final_btn.click(fn=_save_png_to_temp, inputs=[out_final], outputs=[dl_final_file])

            btn_batch.click(
                fn=remove_bg_batch,
                inputs=[multi_imgs, model_sel, smooth, std_on, std_size, std_mode, std_transparent, std_pad_color],
                outputs=[batch_preview, batch_zip, batch_status],
            )
            offset_x_ui.change(fn=replace_bg_step, inputs=[out_enh, rep_sel, rep_color, rep_bg, std_on, std_size, std_mode, std_transparent, std_pad_color, scale_pct_ui, offset_x_ui, offset_y_ui, anchor_ui, clamp_ui], outputs=[out_final, dbg_final])
            offset_y_ui.change(fn=replace_bg_step, inputs=[out_enh, rep_sel, rep_color, rep_bg, std_on, std_size, std_mode, std_transparent, std_pad_color, scale_pct_ui, offset_x_ui, offset_y_ui, anchor_ui, clamp_ui], outputs=[out_final, dbg_final])


        with gr.Tab("Enhancements"):
            with gr.Row():
                with gr.Column():
                    e_in = gr.Image(type="pil")
                    br = gr.Slider(0.5, 1.5, 1.0, label="Brightness"); sh = gr.Slider(0.5, 2.0, 1.0, label="Sharpness"); sa = gr.Slider(0.5, 1.5, 1.0, label="Saturation")
                    e_btn = gr.Button("Preview")
                with gr.Column(): e_out = gr.Image(label="Enhanced Preview")
            e_btn.click(lambda i,b,s,a: ImageEnhance.Brightness(ImageEnhance.Sharpness(ImageEnhance.Color(i.convert("RGB")).enhance(a)).enhance(s)).enhance(b), [e_in, br, sh, sa], e_out)

        with gr.Tab("Model Comparison"):
            with gr.Row():
                comp_in = gr.Image(type="pil"); comp_btn = gr.Button("Run Comparison")
            with gr.Row():
                c1 = gr.Image(label="isnet"); c2 = gr.Image(label="u2net"); c3 = gr.Image(label="silueta")
            comp_btn.click(lambda i: (remove(i, session=sess_isnet), remove(i, session=sess_u2net), remove(i, session=sess_silueta)), comp_in, [c1, c2, c3])

        with gr.Tab("3D Generation"):
            with gr.Row():
                with gr.Column():
                    t_mode = gr.Radio(["Single Image", "Multi-view"], value="Single Image", label="Mode")
                    t_in = gr.Image(type="pil"); t_multi = gr.File(file_count="multiple", visible=False)
                    t_mode.change(lambda m: (gr.update(visible=m=="Single Image"), gr.update(visible=m!="Single Image")), t_mode, [t_in, t_multi])
                    t_btn = gr.Button("GENERATE 3D MODEL (.GLB)", variant="primary")
                    t_cancel = gr.Button("Cancel Generation", variant="stop")
                with gr.Column(): t_view = gr.Model3D(label="3D Preview")
            gen_event = t_btn.click(run_3d_generation, [t_in, t_multi, t_mode], t_view)
            t_cancel.click(fn=None, inputs=None, outputs=None, cancels=[gen_event])

        with gr.Tab("Video Engine"):
            with gr.Row():
                with gr.Column():
                    v_in = gr.Image(type="pil"); v_temp = gr.Radio(["Water Explosion", "Butterfly Oasis", "Studio Glamour", "Forest Mist"], value="Water Explosion", label="Select Effect")
                    v_btn = gr.Button("Generate Cinematic Video", variant="primary")
                    v_cancel = gr.Button("Cancel Rendering", variant="stop")
                with gr.Column(): 
                    v_out = gr.Video(label="Rendered Result")
                    v_status = gr.Textbox(label="Status", interactive=False)
            vid_event = v_btn.click(run_video_engine, [v_in, v_temp], [v_out, v_status])
            v_cancel.click(fn=None, inputs=None, outputs=None, cancels=[vid_event])

        with gr.Tab("Ad Strategy"):
            gr.Markdown("### Aivion Module 4: Content Engine")
            with gr.Row():
                with gr.Column():
                    img_in_strat = gr.Image(type="pil", label="Product Image"); prompt_in_strat = gr.Textbox(label="Instructions", lines=3)
                    plat_in_strat = gr.Dropdown(choices=["Instagram", "TikTok", "All"], value="Instagram"); tone_in_strat = gr.Dropdown(choices=list(TONE_MAP.keys()), value="Casual & Fun")
                    num_in_strat = gr.Slider(1, 5, step=1, value=3, label="Variants"); btn_strat = gr.Button("Generate", variant="primary")
                with gr.Column(): status_strat = gr.Textbox(label="Status", interactive=False); output_md_strat = gr.Markdown("Captions will appear here...")
            btn_strat.click(run_generator, [img_in_strat, prompt_in_strat, plat_in_strat, tone_in_strat, num_in_strat], [status_strat, output_md_strat])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
