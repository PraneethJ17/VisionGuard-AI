# 🛡️ VisionGuard AI

**Integrated Road Scene Understanding and Hazard Analysis System** (educational demo for HackZen 2.0, Computer Vision track)

VisionGuard AI is a Streamlit dashboard that analyses road images and videos and demonstrates many Computer Vision ideas in one place: image processing, object detection, self-supervised visual features, image similarity, vision-language matching, diffusion, promptable segmentation, video tracking and robustness testing.

> **Read this first**
> - The default detector is trained on the **COCO** dataset (80 everyday classes). It has **no pothole, crack or road-damage class**, and this project never claims to detect them.
> - The "scene summary" is a **simple heuristic** (object counts and box sizes). It is **not** a validated hazard or safety system. Do not use it for driving or any safety decision.
> - The app shows **no accuracy, precision, recall or F1 numbers**, because those need labelled ground-truth data that this project does not have.

---

## Contents

1. [Project summary](#1-project-title-and-summary)
2. [Problem statement](#2-problem-statement)
3. [Objectives](#3-project-objectives)
4. [Key features](#4-key-features)
5. [Concepts covered](#5-concepts-covered-from-the-hackzen-20-computer-vision-learning-outline)
6. [Technologies and models](#6-technologies-and-models-used)
7. [Architecture and workflow](#7-system-architecture-and-workflow)
8. [Folder structure](#8-project-folder-structure)
9. [Installation (Windows + VS Code)](#9-installation-on-windows-and-vs-code)
10. [Virtual environment](#10-virtual-environment-setup)
11. [Dependencies](#11-dependency-installation)
12. [Launch command](#12-exact-command-to-launch-the-application)
13. [Using each module](#13-how-to-use-each-dashboard-module)
14. [Model downloads and checkpoints](#14-model-download-and-optional-checkpoint-instructions)
15. [Evaluation metrics explained](#15-explanation-of-evaluation-metrics)
16. [Limitations and failure cases](#16-limitations-and-failure-cases)
17. [Privacy, licensing, ethics](#17-privacy-licensing-and-ethical-considerations)
18. [Future improvements](#18-future-improvements)
19. [GitHub usage](#19-github-usage-instructions)
20. [Demo video outline](#20-demo-video-outline)
21. [Troubleshooting](#21-troubleshooting-common-errors)

---

## 1. Project title and summary

**VisionGuard AI** has a **functional core** that needs only Streamlit, OpenCV, NumPy and Pillow, plus **optional advanced modules** that need extra libraries and model weights. Nothing is downloaded when the app starts. Each heavy model has a **Load** button, and loaded models are cached so they are not loaded twice.

### Implemented vs optional (please read)

| Feature | Status | Needs |
|---|---|---|
| Image upload, validation, safe resizing | **Implemented (core)** | core libraries |
| Grayscale, Gaussian blur, Canny/Sobel edges | **Implemented (core)** | core libraries |
| ViT patch-grid visualisation | **Implemented (core)** | core libraries |
| Forward-diffusion noising demo (NumPy maths) | **Implemented (core)** | core libraries |
| Video sampling + motion-only detection (OpenCV) | **Implemented (core)** | core libraries |
| Robustness: image-statistics comparison | **Implemented (core)** | core libraries |
| YOLO object detection | Implemented, **optional** | `ultralytics`; weights download on Load |
| YOLO + ByteTrack tracking with track IDs | Implemented, **optional** | `ultralytics`, `lapx` |
| Robustness: detection comparison across variants | Implemented, **optional** | `ultralytics` |
| DINOv2 features, PCA map, image similarity | Implemented, **optional** | `torch`, `transformers` (~90 MB) |
| CLIP image-text matching | Implemented, **optional** | `torch`, `transformers` (~600 MB) |
| Segment Anything-style segmentation (point/box) | Implemented, **optional** | `ultralytics` + SAM checkpoint |
| Diffusion inpainting (real diffusion inference) | Implemented, **optional, heavy** | `diffusers`, `torch`, multi-GB model |

"Implemented, optional" means the code exists and is written against each library's documented API. See [section 16](#16-limitations-and-failure-cases) for exactly what has and has not been tested.

## 2. Problem statement

Road cameras, dashcams and street photos contain a lot of information, but turning pixels into understanding needs several different Computer Vision techniques. Learners usually meet these techniques separately. This project asks: **how can one dashboard show, side by side, what each technique does, where it works, and where it fails, on road scenes?**

## 3. Project objectives

- Build one interactive app that integrates the topics of the Computer Vision learning outline.
- Use pretrained models honestly: show what they can detect, and never imply abilities they do not have.
- Keep the app usable on a typical CPU-only Windows laptop, with expensive models strictly optional.
- Show measurements that can really be calculated (counts, timings, consistency), and refuse to invent accuracy figures.
- Make the code readable and beginner-friendly.

## 4. Key features

- Nine-tab dashboard with a sidebar for inputs, model selection and settings.
- File validation (type, size, corrupted files, tiny images, huge images) with friendly messages.
- Per-section error handling: an error in one tab does not crash the others.
- Lazy, cached model loading with clear messages for missing libraries, download failures and low memory.
- Frame-by-frame video processing (the whole video is never held in memory) with frame skipping, a frame limit and a progress bar.
- Honest labelling everywhere (what is a filter, what is a model, what is a heuristic).

## 5. Concepts covered from the HackZen 2.0 Computer Vision learning outline

| Concept | Where in the app |
|---|---|
| Image processing | **Image processing** tab |
| Object detection | **Object detection** tab |
| Self-supervised representations (DINOv2) | **Embeddings & similarity** tab |
| Vision Transformers and patches | Patch grid + DINOv2 PCA map |
| Image similarity | DINOv2 cosine similarity between two images |
| Vision-language models (CLIP) | **Embeddings & similarity** tab, CLIP section |
| Diffusion-based restoration | **Diffusion** tab (forward demo + optional inpainting) |
| Promptable segmentation (SAM) | **Segmentation** tab |
| Video tracking | **Video & tracking** tab |
| Robustness evaluation | **Robustness & evaluation** tab |
| Integrated system design | The whole dashboard (**Home** tab shows the workflow) |

## 6. Technologies and models used

**Libraries**

| Library | Used for | Required? |
|---|---|---|
| Python 3.10 to 3.12 | Language | Yes |
| Streamlit | Web dashboard | Yes |
| OpenCV | Image/video processing, drawing, motion detection | Yes |
| NumPy | Array maths, PCA, diffusion noising | Yes |
| Pillow | Safe image loading | Yes |
| ultralytics | YOLO detection/tracking, SAM | Optional |
| lapx | Linear assignment used by ByteTrack | Optional |
| PyTorch (torch) | Runs the neural networks | Optional |
| transformers | DINOv2 and CLIP | Optional |
| diffusers (+ accelerate) | Diffusion inpainting | Optional |

**Pretrained models (all downloaded on demand; none are trained or fine-tuned by this project)**

| Model | Default identifier | Rough size | Purpose |
|---|---|---|---|
| YOLOv8n / YOLO11n | `yolov8n.pt`, `yolo11n.pt` | about 6 MB | COCO object detection and tracking |
| DINOv2 small | `facebook/dinov2-small` | about 90 MB | Self-supervised ViT features |
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | about 600 MB | Image-text similarity |
| MobileSAM / SAM-B | `mobile_sam.pt`, `sam_b.pt` | about 40 MB / 375 MB | Promptable segmentation |
| Stable Diffusion inpainting | `sd-legacy/stable-diffusion-inpainting` (editable in the app) | multi-gigabyte | Diffusion inpainting |

Sizes are approximate. Model repositories and file names can change; if a download fails, check the model's page and edit the identifier (the inpainting id is editable in the app).

## 7. System architecture and workflow

```
                    ┌────────────────────────────────────────────┐
 Sidebar            │ Upload image / video, resize limit,        │
                    │ YOLO weights, confidence, Load button      │
                    └──────────────────┬─────────────────────────┘
                                       ▼
                    validate (type, size, corrupted) → resize safely
                                       │
     ┌──────────────┬──────────────┬───┴──────────┬──────────────┬─────────────┐
     ▼              ▼              ▼              ▼              ▼             ▼
 Image proc.   Detection      Embeddings/     Diffusion      Segmentation   Video +
 (OpenCV)      (YOLO, opt.)   CLIP (opt.)    (demo + opt.)   (SAM, opt.)    tracking
     │              │              │              │              │             │
     └──────────────┴──────────────┴──────┬───────┴──────────────┴─────────────┘
                                          ▼
                          Robustness & evaluation (counts, timings,
                          consistency, image statistics: no fake accuracy)
```

All code is in `app.py`, organised in clearly commented sections: helpers, image loading, one block of functions per module, then one `render_...` function per dashboard tab. Each tab runs inside `run_safely(...)`, so unexpected errors are shown in that tab only.

## 8. Project folder structure

```
VisionGuardAI/
├── app.py              # the whole application
├── requirements.txt    # core dependencies + commented optional ones
├── .gitignore
└── README.md
```

Model weights that ultralytics downloads appear in the folder from which you launch the app. They are ignored by Git.

## 9. Installation on Windows and VS Code

1. Install **Python 3.10, 3.11 or 3.12** from python.org. Tick **"Add python.exe to PATH"** in the installer.
2. Install **Visual Studio Code** and its **Python** extension.
3. Create a folder, for example `VisionGuardAI`, and put the four project files in it.
4. In VS Code choose **File → Open Folder** and open `VisionGuardAI`.
5. Open a terminal: **Terminal → New Terminal**. Check Python:

```powershell
python --version
```

## 10. Virtual environment setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

- If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` once in that terminal, then activate again. (Or use Command Prompt: `.venv\Scripts\activate.bat`.)
- Your prompt should start with `(.venv)`.
- In VS Code press `Ctrl+Shift+P`, choose **Python: Select Interpreter**, and pick the `.venv` one.

## 11. Dependency installation

**Step A: core (always do this; enough to run the dashboard):**

```powershell
pip install -r requirements.txt
```

**Step B: optional extras (do only what you want).**

*Detection, tracking and segmentation:*

```powershell
pip install ultralytics lapx
```

*DINOv2 and CLIP (CPU-only PyTorch, smaller than the default build):*

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers
```

*Diffusion inpainting (large; see section 14 first):*

```powershell
pip install diffusers accelerate
```

(If you install ultralytics before PyTorch, it installs its own PyTorch, which is fine; it just may be a larger download.)

**GPU:** not required. If you have an NVIDIA GPU and a CUDA build of PyTorch, the optional models use it automatically. The default instructions above are CPU-only.

Never turn off SSL verification to fix installation problems. See [troubleshooting](#21-troubleshooting-common-errors) instead.

## 12. Exact command to launch the application

With the virtual environment active, in the project folder:

```powershell
streamlit run app.py
```

The browser opens at <http://localhost:8501>. Stop the app with `Ctrl + C` in the terminal.

## 13. How to use each dashboard module

First, upload an image in the sidebar (section 1). Videos are uploaded in the same section.

- **🏠 Home:** workflow overview and a live table showing which modules are available on your computer.
- **🖼️ Image processing:** works immediately. Change the blur kernel and sigma, choose Canny or Sobel, and adjust thresholds. Original, grayscale, blurred and edge images appear side by side.
- **🎯 Object detection:** press **Load / switch detector** in the sidebar, then open this tab. Move the confidence slider to see boxes, labels and scores appear or disappear. The tab lists the supported classes and shows a scene summary (heuristic only).
- **🧬 Embeddings & similarity:**
  - The patch grid shows how a Vision Transformer cuts the image into 14 x 14 pixel patches (no model needed).
  - Press **Load DINOv2** to see a PCA colour map of patch features and the embedding. Upload a second image to get a cosine similarity.
  - Press **Load CLIP**, edit the candidate descriptions (one per line), and compare them with the image.
- **🌫️ Diffusion:** the forward-noising demo works immediately (slide *t*). The optional inpainting section needs `diffusers`, a tick-box confirmation, and a multi-gigabyte download.
- **✂️ Segmentation:** choose a checkpoint, press **Load segmentation model**, choose a point (sliders) or a box, and press **Run segmentation**. A SAM checkpoint is **required**; without it nothing is segmented.
- **🎞️ Video & tracking:** choose a mode.
  - *Detection + tracking*: load the tracker weights, then press **Process video**. You get annotated sample frames, a table of unique track IDs per class, and a downloadable video.
  - *Motion only*: no AI model; boxes mark moving regions only.
  - Use **Process every Nth frame** and **Maximum frames** to keep processing quick.
- **🧪 Robustness & evaluation:** set blur, resize and brightness changes. Image statistics always appear. With a detector loaded, press **Run detection on all four versions** to compare counts, timings and consistency.
- **📚 Concepts & limitations:** short explanations of every concept and the project's limits.

## 14. Model download and optional checkpoint instructions

- **Nothing downloads at start-up.** Each model downloads only after you press its Load button (or, for tracking, press the tracker Load button).
- **YOLO weights** (`yolov8n.pt`, `yolo11n.pt`): downloaded by ultralytics into the folder you launched the app from.
- **SAM checkpoint** (`mobile_sam.pt` or `sam_b.pt`): also downloaded by ultralytics on first Load. To use a file you already have, place it in the project folder (the same folder you launch from) before pressing Load.
- **DINOv2 and CLIP:** downloaded from Hugging Face into your user cache (typically under `C:\Users\<you>\.cache\huggingface`).
- **Diffusion inpainting:** the default id is `sd-legacy/stable-diffusion-inpainting`, which is editable in the app. Stable Diffusion models are several gigabytes and need roughly 4 to 8 GB of free RAM. On a CPU, expect minutes per image. Some model repositories require accepting a licence on huggingface.co (and signing in with `huggingface-cli login`); repositories also get moved or removed over time, so verify the id on Hugging Face if loading fails.
- **No internet?** The optional models cannot download. The core features still work.
- If a download fails, the app shows a message instead of crashing. Check your connection, firewall or proxy. Never disable SSL verification.

## 15. Explanation of evaluation metrics

**Shown in the app (these can really be calculated without ground truth):**

| Measurement | Meaning | Caution |
|---|---|---|
| Detection count | Number of boxes above the confidence threshold | A box can be wrong |
| Mean confidence | Average of the model's own scores | Not accuracy |
| Inference time (ms) | Time for one detector call | Depends on your hardware and load |
| Agree / missing / new vs original | Detections matching (same class, IoU ≥ 0.5) the ones on the original image | Measures **consistency**, not correctness |
| Sharpness (Laplacian variance) | Higher = more fine detail | Also affected by noise |
| Mean brightness | Average grey level, 0 to 255 | Says nothing about image quality alone |
| Cosine similarity | How closely two embeddings point the same way | Meaning is relative, no universal cut-off |
| CLIP relative score | Softmax over **your** prompts only | Not a probability; always sums to 100% |

**Defined in the app but deliberately NOT calculated:** precision, recall, F1, mAP and accuracy. They need human-labelled ground truth. For reference: *precision* = correct detections / all detections; *recall* = correct detections / all real objects; *F1* combines the two; *IoU* = overlap area / combined area of two boxes.

## 16. Limitations and failure cases

**What was and was not verified (honest status):**

- Verified by the author's automated checks in a limited environment that had only Python, OpenCV, NumPy and Pillow: the file compiles, and the OpenCV/NumPy functions behave as expected on synthetic images (file validation, filters, edges, drawing, box matching, scene summary, patch grid, PCA colour map, forward-noising maths, and frame-by-frame video processing in motion mode, including graceful rejection of corrupted video).
- **Not run in that environment:** the Streamlit interface itself, YOLO, ByteTrack, DINOv2, CLIP, SAM and diffusion inpainting. Those need libraries and downloads that were not available. The code was written against each library's documented API, but **you should expect to test it on your own computer**, and a library update could require small changes.
- No model was evaluated on labelled data. No performance figures are claimed.

**Known limitations and failure cases:**

- The detector knows only the 80 COCO classes. It **cannot** detect potholes, cracks, lane markings or other road damage.
- Night, rain, fog, glare, motion blur, shadows, small or distant objects, partial occlusion and unusual camera angles can cause **false negatives** (missed objects) and **false positives** (phantom objects).
- The scene summary is a heuristic. A large box may mean "close" or just "big object". It measures no distance and predicts no collision.
- Tracking: IDs can switch or split when objects cross, are hidden, or when many frames are skipped. "Unique IDs" is not a reliable object count.
- Motion-only mode reacts to anything that moves (including camera shake and moving shadows) and knows nothing about object types.
- CLIP and DINOv2 scores are relative and can mislead on fine details or rare scenes.
- Diffusion inpainting invents plausible content. It must never be used as evidence of what was really in a scene.
- The annotated video may use the `mp4v` codec when H.264 is unavailable; many browsers cannot play that inline, so download it instead.
- Large images are shrunk and videos are limited (200 MB, adjustable frame limit), so results describe the processed version.
- CPU processing is slow for big inputs; the optional models can use gigabytes of RAM.

## 17. Privacy, licensing and ethical considerations

- **Privacy:** the app runs on your computer. Your images and videos are not uploaded anywhere by this code. Only model weights are downloaded. Road images can contain faces, number plates and locations; use only media you are allowed to use, do not commit such media to GitHub, and blur or remove identifying details before sharing screenshots or demo videos.
- **Ethics:** do not use this project for surveillance of individuals, for driving decisions, or for any safety-critical purpose. Detection models can perform unevenly across conditions and populations; report their limits honestly.
- **Licensing (verify on each official page before redistributing; this is not legal advice):** Ultralytics YOLO is released under AGPL-3.0, with a separate enterprise licence available, and AGPL has obligations if you distribute or run modified versions as a network service. DINOv2, CLIP, SAM/MobileSAM and Stable Diffusion have their own licences (for example, Stable Diffusion models use a CreativeML OpenRAIL licence with use restrictions). Check each model card. Add a licence for your own code (for example MIT) if you publish it.
- **Honest claims:** this project does not claim to be unique, validated or production-ready.

## 18. Future improvements

- Evaluate on a **labelled** road dataset and report precision, recall and mAP properly.
- Fine-tune or test a model on road-damage data **before** claiming any pothole detection.
- Add click-on-image prompts for segmentation (currently sliders keep dependencies minimal).
- Add depth estimation or camera calibration for real distance estimates.
- Add lane detection and traffic-sign recognition trained on suitable data.
- Add batch processing, result export (CSV) and comparison across models.
- Add automated tests for the optional modules on machines where they are installed.

## 19. GitHub usage instructions

1. Create a free account at github.com and click **New repository**. Name it `VisionGuardAI`. Leave it empty (no README) and create it.
2. In the VS Code terminal, inside the project folder:

```powershell
git init
git add .
git commit -m "Initial commit: VisionGuard AI"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/VisionGuardAI.git
git push -u origin main
```

3. If Git is not installed, get it from git-scm.com. If Git asks who you are:

```powershell
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
```

4. `.gitignore` already excludes the virtual environment, model weights, videos and secrets. Run `git status` before pushing and check that no personal images or videos are listed.
5. Add a licence file and your team members' names to this README before submission.

## 20. Demo video outline

Target: 3 to 5 minutes. Record with `Win + G` (Xbox Game Bar) or OBS Studio.

| Time | Content |
|---|---|
| 0:00 | Title, goal, and the honest scope statement (no pothole detection, no accuracy claims) |
| 0:30 | Home tab: workflow and the module status table (core vs optional) |
| 1:00 | Upload an image; Image processing: grayscale, blur, edges, sliders |
| 1:45 | Load detector; Object detection: confidence slider, supported classes, scene summary |
| 2:30 | Embeddings: patch grid, DINOv2 map, CLIP prompts (say the scores are not probabilities) |
| 3:15 | Diffusion forward demo; (optional) segmentation with a point prompt |
| 3:45 | Video: frame skipping, track IDs, mention ID switches |
| 4:30 | Robustness: blur/resize/brightness comparison; explain why there are no accuracy numbers |
| 5:00 | Limitations, ethics, future work |

Only demonstrate modules that actually ran on your computer. If an optional module is not installed, say so and show the message the app displays.

## 21. Troubleshooting common errors

| Problem | What to do |
|---|---|
| `python` is not recognised | Reinstall Python and tick "Add python.exe to PATH", then reopen the terminal |
| PowerShell blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use Command Prompt |
| `streamlit` is not recognised | Activate the virtual environment first; then `pip install -r requirements.txt` |
| Optional tab says a library is missing | Install it with the command shown in the tab (see section 11), then restart the app |
| `DLL load failed` when importing torch | Install the latest "Microsoft Visual C++ Redistributable" (x64) and try again; use Python 3.10 to 3.12 |
| Model download fails | Check internet, firewall or proxy; try another network (a mobile hotspot), and check the PC date and time. Do **not** disable SSL verification. |
| Hugging Face error about access or a missing repository | The model may be gated or moved. Accept its licence on huggingface.co, run `huggingface-cli login`, or edit the model id |
| `MemoryError` or "paging file too small" | Close other programs, use smaller models, or skip diffusion |
| First tracking run tries to install `lap` | Run `pip install lapx` yourself before starting the app |
| Annotated video will not play in the browser | The `mp4v` codec is not browser-friendly; use the download button and a desktop player |
| "OpenCV could not open this video" | Convert the video to a standard H.264 MP4 and try again |
| Upload rejected as too large | Streamlit's default limit is 200 MB; use a shorter clip, or launch with `streamlit run app.py --server.maxUploadSize 500` |
| Port 8501 already in use | `streamlit run app.py --server.port 8502` |
| Everything is slow | Lower the max image side, use a smaller inference size, increase "Process every Nth frame" |
| Detector finds nothing | Lower the confidence threshold; remember objects outside the 80 COCO classes can never be detected |

If something fails, copy the **full** error message from the terminal; it is the most useful clue.
