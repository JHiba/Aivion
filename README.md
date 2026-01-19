# Aivion
**Aivion streamlines digital commerce by automating high-quality visual content creation for small businesses.**

## Phase 1: Background Removal
In this phase, we implemented an automated pipeline to process raw product images. The system detects the subject and cleanly removes the background, enhances it, replaces it background and prepares it for 3D reconstruction.

![bag-2661412_1280](https://github.com/user-attachments/assets/51b335a8-6759-42a5-a6ea-8d06faf78cb1)
### Output Example
<img width="1024" height="1024" alt="bg_removed" src="https://github.com/user-attachments/assets/89bacacf-f7f9-40b2-b198-c60c6dfabf16" />


---

## Phase 2: 3D Visualization (Trellis)
Using the clean images from Phase 1, we utilize the Trellis model to generate 3D assets. This allows users to view the product from all angles.

### Demo Video

3D visualization of a bag and a mug, using pictures given at run time.

https://github.com/user-attachments/assets/8269455c-4308-4235-a66b-4d6cd507ca54


https://github.com/user-attachments/assets/142fbc24-3cfb-45db-aedc-f6f3b53884c6

## Requirements & Environment

### Phase 1: Background Removal (Local)
These libraries are required to run the background removal script locally:
* `onnxruntime`
* `pillow`
* `numpy`
* `rembg`

### Phase 2: 3D Visualization (Trellis)
**Hardware Note:** The Trellis model requires high-end GPU resources. We utilized **Hyperstack Cloud GPU** to execute the inference and generation pipeline.

**Dependencies:**
* `torch` (CUDA version recommended)
* *See `trellis_ui.py` imports for UI libraries (likely `gradio` or `streamlit`)*
