# # bg_removal.py 
# import gradio as gr
# from rembg import remove, new_session
# from PIL import Image, ImageEnhance, ImageOps, Image
# import numpy as np
# from scipy.ndimage import gaussian_filter
# import tempfile, time, os, zipfile, io
# from typing import Any, Optional, Tuple, List

# # =========================
# # Model sessions (preload)
# # =========================
# session_u2net = new_session("u2net")
# session_isnet = new_session("isnet-general-use")
# session_silueta = new_session("silueta")

# # =========================
# # Utilities
# # =========================
# def parse_color(color_str):
#     if not color_str:
#         return (255, 255, 255, 255)
#     s = color_str.strip()
#     if s.startswith("#"):
#         s = s.lstrip("#")
#         if len(s) == 6:
#             r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16)
#             return (r, g, b, 255)
#         if len(s) == 8:
#             r = int(s[0:2], 16); g = int(s[2:4], 16); b = int(s[4:6], 16); a = int(s[6:8], 16)
#             return (r, g, b, a)
#     if s.startswith("rgba") or s.startswith("rgb"):
#         inside = s[s.find("(")+1:s.rfind(")")]
#         parts = [p.strip() for p in inside.split(",")]
#         try:
#             r = int(float(parts[0])); g = int(float(parts[1])); b = int(float(parts[2]))
#             if len(parts) == 4:
#                 a = float(parts[3]); a = int(a * 255) if a <= 1 else int(a)
#             else:
#                 a = 255
#             return (r, g, b, a)
#         except Exception:
#             return (255, 255, 255, 255)
#     return (255, 255, 255, 255)

# def _auto_white_balance_gray_world(img_rgb_np):
#     eps = 1e-6
#     r, g, b = img_rgb_np[..., 0], img_rgb_np[..., 1], img_rgb_np[..., 2]
#     mean_r = r.mean() + eps
#     mean_g = g.mean() + eps
#     mean_b = b.mean() + eps
#     mean_gray = (mean_r + mean_g + mean_b) / 3.0
#     scale_r = mean_gray / mean_r
#     scale_g = mean_gray / mean_g
#     scale_b = mean_gray / mean_b
#     out = img_rgb_np.astype(np.float32).copy()
#     out[..., 0] = np.clip(out[..., 0] * scale_r, 0, 255)
#     out[..., 1] = np.clip(out[..., 1] * scale_g, 0, 255)
#     out[..., 2] = np.clip(out[..., 2] * scale_b, 0, 255)
#     return out.astype(np.uint8)

# def _apply_temperature(img_rgb_np, temp_value):
#     if temp_value == 0:
#         return img_rgb_np
#     max_scale_delta = 0.10
#     delta = (temp_value / 50.0) * max_scale_delta
#     r_scale = 1.0 + delta
#     b_scale = 1.0 - delta
#     out = img_rgb_np.astype(np.float32).copy()
#     out[..., 0] = np.clip(out[..., 0] * r_scale, 0, 255)
#     out[..., 2] = np.clip(out[..., 2] * b_scale, 0, 255)
#     return out.astype(np.uint8)

# def enhanced_preprocess_image(
#     image, do_resize=True, brightness=1.0, sharpness=1.0,
#     saturation=1.0, auto_wb=False, temperature=0
# ):
#     img = image.convert("RGBA")
#     rgb = img.convert("RGB")

#     if auto_wb:
#         arr = np.array(rgb); arr = _auto_white_balance_gray_world(arr)
#         rgb = Image.fromarray(arr, mode="RGB")
#     if temperature != 0:
#         arr = np.array(rgb); arr = _apply_temperature(arr, temperature)
#         rgb = Image.fromarray(arr, mode="RGB")

#     if saturation != 1.0:
#         rgb = ImageEnhance.Color(rgb).enhance(float(saturation))
#     if brightness != 1.0:
#         rgb = ImageEnhance.Brightness(rgb).enhance(float(brightness))
#     if sharpness != 1.0:
#         rgb = ImageEnhance.Sharpness(rgb).enhance(float(sharpness))

#     img = rgb.convert("RGBA")
#     img = ImageEnhance.Contrast(img).enhance(1.15)

#     if do_resize and max(img.size) > 1024:
#         img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
#     return img

# def smooth_edges_only(image, blur_amount=1):
#     if blur_amount == 0:
#         return image
#     img = image.convert("RGBA")
#     arr = np.array(img)
#     if arr.shape[2] < 4:
#         return image
#     alpha = arr[:, :, 3].astype(np.float32)
#     alpha_blurred = gaussian_filter(alpha, sigma=blur_amount)
#     alpha_blurred = np.clip(alpha_blurred, 0, 255).astype(np.uint8)
#     arr[:, :, 3] = alpha_blurred
#     return Image.fromarray(arr)

# def ensure_alpha(removed_img):
#     img = removed_img.convert("RGBA")
#     arr = np.array(img)
#     if arr.shape[2] == 4:
#         alpha = arr[:, :, 3]
#         if alpha.min() < 255:
#             return img, "native_alpha"

#     rgb = arr[:, :, :3].astype(np.float32)
#     h, w = rgb.shape[:2]
#     pad = max(4, int(min(h, w) * 0.02))
#     samples = np.concatenate([
#         rgb[:pad, :, :].reshape(-1, 3),
#         rgb[-pad:, :, :].reshape(-1, 3),
#         rgb[:, :pad, :].reshape(-1, 3),
#         rgb[:, -pad:, :].reshape(-1, 3)
#     ], axis=0)
#     bg_color = np.median(samples, axis=0)
#     dist = np.sqrt(((rgb - bg_color) ** 2).sum(axis=2))
#     med = np.median(dist); thr = float(max(20.0, med * 1.2))
#     mask = (dist > thr).astype(np.uint8) * 255
#     mask = gaussian_filter(mask.astype(np.float32), sigma=2.0)
#     mask = np.clip(mask, 0, 255).astype(np.uint8)
#     arr[:, :, 3] = mask
#     return Image.fromarray(arr), "computed_mask"

# # =========================
# # Standardize size / AR
# # =========================
# def _letterbox_rgba(img_rgba, size, pad_rgba=(0, 0, 0, 0)):
#     img = img_rgba.convert("RGBA")
#     w, h = img.size
#     if w == 0 or h == 0:
#         return Image.new("RGBA", (size, size), pad_rgba)
#     scale = min(size / w, size / h)
#     new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#     resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#     canvas = Image.new("RGBA", (size, size), pad_rgba)
#     off_x = (size - new_w) // 2; off_y = (size - new_h) // 2
#     canvas.alpha_composite(resized, dest=(off_x, off_y))
#     return canvas

# def _cover_rgba(img_rgba, size):
#     img = img_rgba.convert("RGBA")
#     w, h = img.size
#     if w == 0 or h == 0:
#         return Image.new("RGBA", (size, size), (0, 0, 0, 0))
#     scale = max(size / w, size / h)
#     new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
#     resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#     left = (new_w - size) // 2; top = (new_h - size) // 2
#     right = left + size; bottom = top + size
#     return resized.crop((left, top, right, bottom))

# def standardize_canvas(img_rgba, size, mode, pad_rgba):
#     if size is None or size <= 0:
#         return img_rgba.convert("RGBA")
#     if "Letterbox" in mode:
#         return _letterbox_rgba(img_rgba, size, pad_rgba)
#     return _cover_rgba(img_rgba, size)

# # --- download helper ---------------------------------------------------------
# def _save_png_to_temp(image: Image.Image, prefix: str = "aivion") -> Optional[str]:
#     if image is None:
#         return None
#     image = image.convert("RGBA")
#     tmpdir = tempfile.gettempdir()
#     ts = int(time.time() * 1000)
#     path = os.path.join(tmpdir, f"{prefix}_{ts}.png")
#     image.save(path, "PNG")
#     return path
# # -----------------------------------------------------------------------------


# # =========================
# # Background removal (Enhanced core)
# # =========================
# def remove_bg_enhanced(
#     input_img, model_choice, smoothing, apply_preprocess,
#     brightness, sharpness, saturation, auto_wb, temperature,
#     std_on, std_size, std_mode, std_transparent, std_pad_color
# ):
#     if input_img is None:
#         return None, "No input image provided."

#     img = input_img.convert("RGBA")
#     if apply_preprocess:
#         img = enhanced_preprocess_image(
#             img, do_resize=True, brightness=brightness, sharpness=sharpness,
#             saturation=saturation, auto_wb=auto_wb, temperature=int(temperature)
#         )

#     if "isnet-general-use" in model_choice:
#         out = remove(img, session=session_isnet)
#     elif "silueta" in model_choice:
#         out = remove(img, session=session_silueta)
#     else:
#         out = remove(img, session=session_u2net)

#     if smoothing and smoothing > 0:
#         out = smooth_edges_only(out, smoothing)

#     if std_on:
#         pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
#         out = standardize_canvas(out, int(std_size), std_mode, pad_rgba)

#     out_checked, flag = ensure_alpha(out)
#     arr = np.array(out_checked.convert("RGBA")); alpha = arr[:, :, 3]
#     debug = (
#         f"[ENHANCED] Mode: {out_checked.mode} | Size: {out_checked.size} | "
#         f"Alpha min/max: {int(alpha.min())}/{int(alpha.max())} | "
#         f"alpha_source: {flag} | standardized: {bool(std_on)}"
#     )
#     return out_checked, debug

# def replace_bg_step(
#     removed_img, replace_option, color_choice, bg_upload,
#     std_on, std_size, std_mode, std_transparent, std_pad_color
# ):
#     if removed_img is None:
#         return None, "No removed image available. Run Remove Background first."

#     fg, flag = ensure_alpha(removed_img)
#     w, h = fg.size

#     if replace_option is None or replace_option == "No Replacement":
#         composed = fg; debug = f"No replacement applied. alpha_source: {flag}"
#     elif replace_option == "Solid Color":
#         rgba = parse_color(color_choice)
#         bg = Image.new("RGBA", (w, h), rgba)
#         composed = Image.alpha_composite(bg, fg)
#         debug = f"Applied solid color {rgba} | alpha_source: {flag}"
#     elif replace_option == "Upload Image":
#         if bg_upload is None:
#             return fg, "No background image uploaded."
#         bg = bg_upload.convert("RGBA")
#         bg_fitted = ImageOps.fit(bg, (w, h), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
#         composed = Image.alpha_composite(bg_fitted, fg)
#         debug = f"Applied uploaded background (fit to {w}x{h}) | alpha_source: {flag}"
#     else:
#         composed = fg; debug = "Unknown option"

#     if std_on:
#         pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
#         composed = standardize_canvas(composed, int(std_size), std_mode, pad_rgba)
#         debug += f" | standardized: True -> {std_size}px {std_mode}"

#     return composed, debug

# def test_all_models(image):
#     if image is None:
#         return None, None, None
#     img = image.convert("RGBA")
#     out_u2net = out_isnet = out_silueta = None
#     try: out_u2net = remove(img, session=session_u2net)
#     except Exception: out_u2net = None
#     try: out_isnet = remove(img, session=session_isnet)
#     except Exception: out_isnet = None
#     try: out_silueta = remove(img, session=session_silueta)
#     except Exception: out_silueta = None
#     return out_u2net, out_isnet, out_silueta

# # ----- ENHANCEMENT-ONLY PREVIEW -----
# def enhance_only_preview(
#     input_img, brightness, sharpness, saturation, auto_wb, temperature,
#     std_on, std_size, std_mode, std_transparent, std_pad_color
# ):
#     if input_img is None:
#         return None, "No input image provided."
#     img = enhanced_preprocess_image(
#         input_img, do_resize=False,
#         brightness=brightness, sharpness=sharpness, saturation=saturation,
#         auto_wb=auto_wb, temperature=int(temperature)
#     )
#     if std_on:
#         pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
#         img = standardize_canvas(img, int(std_size), std_mode, pad_rgba)
#     dbg = f"[PREVIEW] Size: {img.size} | std: {bool(std_on)}"
#     return img, dbg

# # ===== Robust batch file handling =====
# def _load_image_from_any(file_obj: Any) -> Tuple[Optional[Image.Image], str]:
#     """
#     Try to load a PIL Image from any Gradio File shape:
#     - str path
#     - dict with 'path' or 'name' (and maybe 'data' bytes)
#     - object with .path or .name
#     - dict with 'data' (bytes)
#     Returns (image_or_None, human_readable_name)
#     """
#     # 1) string path
#     if isinstance(file_obj, str):
#         name_show = os.path.basename(file_obj)
#         if os.path.exists(file_obj):
#             return Image.open(file_obj).convert("RGBA"), name_show
#         return None, name_show

#     # 2) dict-like
#     if isinstance(file_obj, dict):
#         data = file_obj.get("data", None)
#         if data is not None:
#             try:
#                 return Image.open(io.BytesIO(data)).convert("RGBA"), file_obj.get("orig_name") or file_obj.get("name") or "uploaded"
#             except Exception:
#                 pass
#         path = file_obj.get("path") or file_obj.get("name")
#         if path and os.path.exists(path):
#             return Image.open(path).convert("RGBA"), os.path.basename(path)
#         return None, file_obj.get("orig_name") or file_obj.get("name") or "file"

#     # 3) object with attributes
#     path = getattr(file_obj, "path", None) or getattr(file_obj, "name", None)
#     if path and isinstance(path, str) and os.path.exists(path):
#         return Image.open(path).convert("RGBA"), os.path.basename(path)

#     return None, "file"

# # ----- BATCH REMOVAL ---------------------------------------------------------
# def remove_bg_batch(
#     input_files, model_choice, smoothing,
#     std_on, std_size, std_mode, std_transparent, std_pad_color
# ):
#     """
#     input_files: list of mixed shapes from gr.File (multiple)
#     Returns: (list_of_images_for_gallery, path_to_zip, status_message)
#     """
#     if not input_files:
#         return [], None, "No files provided."

#     input_files = input_files[:4]
#     results: List[Image.Image] = []
#     msgs = []

#     tmpdir = tempfile.gettempdir()
#     zip_path = os.path.join(tmpdir, f"aivion_batch_{int(time.time())}.zip")
#     zf = zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED)

#     for idx, f in enumerate(input_files, start=1):
#         img, display_name = _load_image_from_any(f)
#         if img is None:
#             msgs.append(f"[{idx}] Skipped: couldn’t read {display_name}")
#             continue

#         try:
#             if "isnet-general-use" in model_choice:
#                 out = remove(img, session=session_isnet)
#             elif "silueta" in model_choice:
#                 out = remove(img, session=session_silueta)
#             else:
#                 out = remove(img, session=session_u2net)

#             if smoothing and smoothing > 0:
#                 out = smooth_edges_only(out, smoothing)

#             if std_on:
#                 pad_rgba = (0, 0, 0, 0) if std_transparent else parse_color(std_pad_color)
#                 out = standardize_canvas(out, int(std_size), std_mode, pad_rgba)

#             stem = os.path.splitext(os.path.basename(display_name))[0] or f"image_{idx}"
#             out_path = os.path.join(tmpdir, f"{stem}_removed.png")
#             out.save(out_path, "PNG")
#             zf.write(out_path, arcname=os.path.basename(out_path))

#             results.append(out)
#             msgs.append(f"[{idx}] OK: {display_name} → removed")
#         except Exception as e:
#             msgs.append(f"[{idx}] Error on {display_name}: {e}")

#     zf.close()

#     zip_out = zip_path if len(results) > 0 else None
#     status = "\n".join(msgs) if msgs else "Done."
#     return results, zip_out, status
# # -----------------------------------------------------------------------------


# # =========================
# # UI (reorganized)
# # =========================
# css = """
# body {background: #ffffff; color: #111; font-family:'Segoe UI', Tahoma, sans-serif;}
# .gradio-container {max-width:1200px; margin:12px auto;}
# h1, h2, h3 {color:#000000 !important; font-weight:600;}
# label, .gr-text-input, .gr-button {color:#111;}
# .gr-image {border-radius:12px; box-shadow: 0 6px 20px rgba(0,0,0,0.15);}
# .small-note {font-size: 12px; color:#444;}
# """

# with gr.Blocks(css=css, title="Aivion — Enhanced Background Removal") as demo:
#     gr.Markdown("#  AIVION")
#     gr.Markdown("Tab 1: **Removal & Replacement** · Tab 2: **Enhancements (Preview Only)** · Plus Model Comparison & About.")

#     # ----------------------- TAB 1: Removal & Replacement -----------------------
#     with gr.Tab(" Background Removal & Replacement"):
#         with gr.Row():
#             with gr.Column(scale=1):
#                 # Single-image mode
#                 in_img = gr.Image(type="pil", label="Upload Single Product Image")

#                 # Multi-image mode
#                 multi_imgs = gr.File(file_count="multiple", file_types=["image"], label="📁 Or upload multiple images (2–4 recommended)")

#                 model_sel = gr.Radio(
#                     choices=[
#                         "isnet-general-use (Recommended for Products)",
#                         "u2net (Default)",
#                         "silueta (Objects)"
#                     ],
#                     value="isnet-general-use (Recommended for Products)",
#                     label="Choose Model"
#                 )
#                 smooth = gr.Slider(minimum=0, maximum=3, value=0, step=0.5, label="Edge Smoothing")

#                 # Standardization for outputs
#                 gr.Markdown("### Output Size")
#                 std_on = gr.Checkbox(label="Standardize output size (square)", value=True)
#                 with gr.Row():
#                     std_size = gr.Dropdown(choices=[512, 768, 1024], value=1024, label="Target size (px)")
#                     std_mode = gr.Radio(choices=["Letterbox (pad)", "Cover (crop)"], value="Letterbox (pad)", label="Fill mode")
#                 with gr.Row():
#                     std_transparent = gr.Checkbox(label="Pad with transparency (RGBA)", value=True)
#                     std_pad_color = gr.ColorPicker(label="Pad color (when transparency is OFF)", value="#ffffff")

#                 # Buttons
#                 btn_enh = gr.Button("🗡 Remove Background (Single)", variant="primary")
#                 gr.Markdown("---")
#                 gr.Markdown("###  Batch Remove Backgrounds")
#                 btn_batch = gr.Button(" Remove Backgrounds (Batch)", variant="primary")

#             with gr.Column(scale=1):
#                 # Single image outputs
#                 gr.Markdown("### Removed (transparent) — Single")
#                 out_enh = gr.Image(type="pil", label="Removed Image", elem_classes="gr-image")
#                 dbg_enh = gr.Textbox(label="Debug", interactive=False)

#                 # Download removed
#                 dl_removed_btn = gr.Button("⬇️ Download Removed PNG")
#                 dl_removed_file = gr.File(label="Download: removed.png")

#                 gr.Markdown("### Replacement (Single)")
#                 rep_sel = gr.Radio(choices=["No Replacement", "Solid Color", "Upload Image"],
#                                    value="No Replacement", label="Replace with")
#                 rep_color = gr.ColorPicker(label="Pick solid color (if chosen)", value="#ffffff")
#                 rep_bg = gr.Image(type="pil", label="Upload background image (if chosen)")
#                 btn_replace = gr.Button("🖌 Apply Replacement", variant="secondary")

#                 gr.Markdown("### Final Result (Single)")
#                 out_final = gr.Image(type="pil", label="Final Result", elem_classes="gr-image")
#                 dbg_final = gr.Textbox(label="Replacement debug info", interactive=False)

#                 # Download final
#                 dl_final_btn = gr.Button("⬇️ Download Final PNG")
#                 dl_final_file = gr.File(label="Download: final.png")

#                 gr.Markdown("---")
#                 # Batch outputs
#                 gr.Markdown("### Batch Results (Preview)")
#                 batch_preview = gr.Gallery(label="Batch Preview", columns=2, height="auto")
#                 batch_zip = gr.File(label="⬇️ Download All (ZIP)")
#                 batch_status = gr.Textbox(label="Batch Status", interactive=False)

#         # Single Removal (no enhancements in Tab 1)
#         btn_enh.click(
#             fn=remove_bg_enhanced,
#             inputs=[
#                 in_img, model_sel, smooth, gr.State(False),
#                 gr.State(1.0), gr.State(1.0), gr.State(1.0), gr.State(False), gr.State(0),
#                 std_on, std_size, std_mode, std_transparent, std_pad_color
#             ],
#             outputs=[out_enh, dbg_enh]
#         )

#         # Replacement for single result
#         btn_replace.click(
#             fn=replace_bg_step,
#             inputs=[
#                 out_enh, rep_sel, rep_color, rep_bg,
#                 std_on, std_size, std_mode, std_transparent, std_pad_color
#             ],
#             outputs=[out_final, dbg_final]
#         )

#         # Downloads (single)
#         dl_removed_btn.click(fn=_save_png_to_temp, inputs=[out_enh], outputs=[dl_removed_file])
#         dl_final_btn.click(fn=_save_png_to_temp, inputs=[out_final], outputs=[dl_final_file])

#         # Batch removal (gallery, zip, status)
#         btn_batch.click(
#             fn=remove_bg_batch,
#             inputs=[multi_imgs, model_sel, smooth, std_on, std_size, std_mode, std_transparent, std_pad_color],
#             outputs=[batch_preview, batch_zip, batch_status]
#         )

#     # ----------------------- TAB 2: Enhancements (Preview Only) -------------
#     with gr.Tab(" Enhancements (Preview Only)"):
#         with gr.Row():
#             with gr.Column(scale=1):
#                 in_prev = gr.Image(type="pil", label="Upload Product Image")

#                 # Enhancements
#                 gr.Markdown("### Adjustments")
#                 brightness_p = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, step=0.05, label="Brightness")
#                 sharpness_p  = gr.Slider(minimum=0.5, maximum=2.0, value=1.0, step=0.05, label="Sharpness / Clarity")
#                 saturation_p = gr.Slider(minimum=0.5, maximum=1.5, value=1.0, step=0.05, label="Color (Saturation)")
#                 auto_wb_p    = gr.Checkbox(label="Auto White Balance (Gray-World)", value=False)
#                 temp_p       = gr.Slider(minimum=-50, maximum=50, value=0, step=1, label="Temperature (Cool ↔ Warm)")

#                 gr.Markdown("###  Output Size (optional)")
#                 std_on_p = gr.Checkbox(label="Standardize output size (square)", value=False)
#                 with gr.Row():
#                     std_size_p = gr.Dropdown(choices=[512, 768, 1024], value=1024, label="Target size (px)")
#                     std_mode_p = gr.Radio(choices=["Letterbox (pad)", "Cover (crop)"], value="Letterbox (pad)", label="Fill mode")
#                 with gr.Row():
#                     std_transparent_p = gr.Checkbox(label="Pad with transparency (RGBA)", value=True)
#                     std_pad_color_p = gr.ColorPicker(label="Pad color (when transparency is OFF)", value="#ffffff")

#                 btn_preview = gr.Button(" Preview Enhancements", variant="secondary")

#                 # Download preview
#                 dl_preview_btn = gr.Button("⬇️ Download Preview PNG")
#                 dl_preview_file = gr.File(label="Download: preview.png")

#             with gr.Column(scale=1):
#                 prev_img = gr.Image(type="pil", label="Enhanced Preview", elem_classes="gr-image")
#                 prev_dbg = gr.Textbox(label="Preview debug", interactive=False)

#         # Generate preview (no removal here)
#         btn_preview.click(
#             fn=enhance_only_preview,
#             inputs=[
#                 in_prev, brightness_p, sharpness_p, saturation_p, auto_wb_p, temp_p,
#                 std_on_p, std_size_p, std_mode_p, std_transparent_p, std_pad_color_p
#             ],
#             outputs=[prev_img, prev_dbg]
#         )
#         dl_preview_btn.click(fn=_save_png_to_temp, inputs=[prev_img], outputs=[dl_preview_file])

#     # ----------------------- Model Comparison -----------------------
#     with gr.Tab(" Model Comparison"):
#         gr.Markdown("Compare the three models side-by-side")
#         with gr.Row():
#             comp_input = gr.Image(type="pil", label="Upload Product Image")
#             compare_btn = gr.Button(" Run Comparison")
#         with gr.Row():
#             with gr.Column():
#                 gr.Markdown("### u2net (Default)")
#                 cmp_u2 = gr.Image(type="pil", elem_classes="gr-image")
#             with gr.Column():
#                 gr.Markdown("### isnet-general-use")
#                 cmp_is = gr.Image(type="pil", elem_classes="gr-image")
#             with gr.Column():
#                 gr.Markdown("### silueta")
#                 cmp_si = gr.Image(type="pil", elem_classes="gr-image")
#         compare_btn.click(fn=test_all_models, inputs=[comp_input], outputs=[cmp_u2, cmp_is, cmp_si])

#     # ----------------------- About -----------------------
#     with gr.Tab("ℹ️ About"):
#         gr.Markdown("""
# # Best Practices for Product Images:
# -  Keep your camera steady  
# -  Ensure **good lighting**  
# -  Avoid shadows and cluttered backgrounds  
# -  Center your product in the frame  

# ##  Model Guide:
# -  **isnet-general-use** → Best for most e-commerce products  
# -  **u2net** → Fast, general purpose  
# -  **silueta** → Works for simple, clear objects  

# ##  Settings:
# -  **Edge Smoothing**: 0.5–1.0 for softer edges  
# -  **No Smoothing**: Keep 0 for sharp cutouts  

# **Workflow**
# 1. Tab 1: Remove background (single or batch) → (optional) replace & standardize → download.  
# 2. Tab 2: Tune enhancements → preview → (go back to Tab 1 for removal).
#         """)
#     gr.HTML("<div style='text-align:center; padding:12px; color:#999;'>© 2025 Aivion</div>")

# if __name__ == "__main__":
#     # For VS Code local dev, consider:
#     # demo.launch(server_name="0.0.0.0", server_port=7860)
#     demo.launch(share=True)
