# Aivion: Orchestrated AI Content Engine

Aivion is a high-performance digital commerce automation platform that generates high-quality visual content (Video & 3D) and marketing strategies using a distributed "Hub-and-Spoke" architecture.

## 🏗️ System Architecture
The platform is designed for scalability, separating the orchestration logic from the heavy generative engines:

* **The Hub (`app1.py`):** The central orchestrator and Gradio UI. It handles user inputs and manages the Ad Strategist (Llama-3/Groq).
* **Video Engine Spoke (`video2.py`):** Dedicated script for high-fidelity video generation using **CogVideoX**. Optimized for NVIDIA RTX A6000 (48GB VRAM).
* **3D Engine Spoke (`trellis_worker.py`):** Specialized worker for generating 3D assets using the **TRELLIS** framework.

## 🚀 Key Features
- **AI Ad Strategist:** Generates data-driven marketing copy and engagement plans.
- **Cinematic Video Generation:** Produces high-resolution product videos from text prompts.
- **3D Asset Creation:** Transforms concepts into 3D models for immersive commerce.
- The 3D engine requires the TRELLIS framework:
1. Clone the TRELLIS repository.
2. Follow the TRELLIS installation guide to set up the environment.
- **VRAM Optimization:** Intelligent memory management to run large-scale models on a single GPU.

## 🛠️ Tech Stack
- **Languages:** Python
- **Frameworks:** Gradio, PyTorch
- **Models:** CogVideoX (Video), TRELLIS (3D), Llama-3/Gemini (LLMs)
- **Infrastructure:** NVIDIA RTX A6000 via Hyperstack Cloud

## 📦 Installation & Setup
1. Clone the repository:
   ```bash
   git clone [https://github.com/JHiba/Aivion.git](https://github.com/JHiba/Aivion.git)
   cd Aivion

Install dependencies:
pip install -r requirements.txt

Set Environment Variables:
export GROQ_API_KEY="your_key"
export GEMINI_API_KEY="your_key"

Run the Hub:
python app1.py

## 📺 Live Demo
Click the badge below to watch the Aivion platform in action, featuring the orchestrated video generation and 3D vision pipeline.

[![Watch Demo](https://img.shields.io/badge/Watch-Live_Demo-red?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/file/d/1lHQ8OIkOPmaD1ycq5ogGNijHDPm67Hh4/view?usp=sharing)


Focused on scalable AI Architecture.

