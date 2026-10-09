# Sketch2CAD 

Sketch2CAD is a web application that takes a hand-drawn pencil sketch, automatically straightens it, and enhances it into a crisp, noise-free digital image—all using pure computer vision (no ML models).

## Features
- **Phase 1: Enhancement Pipeline**
  - Pure Image Processing: Uses morphological operations to remove paper brightness variations.
  - Stroke Extraction: Black-hat morphology isolates thin pencil strokes.
  - Percentile Normalization: Cleans background noise safely.
  
- **Phase 2: Page Detection & Straightening**
  - **Auto-Detection**: Scans the image to find the bounds of the paper (using edge, brightness, or grabcut strategies).
  - **Interactive Editor**: Review and adjust the detected corners with an on-screen magnifier (loupe).
  - **Perspective Correction**: Un-skews the image and can optionally force it to standard A4 or Letter aspect ratios.

## Tech Stack
- **Backend**: Python 3.10+, FastAPI, OpenCV, NumPy, Pillow
- **Frontend**: Vanilla HTML/CSS/JS

## Setup

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

3. Run the application:
   ```bash
   python backend/main.py
   ```

4. Open your browser and navigate to `http://localhost:8000`.

## API

- `POST /api/detect-page`: Analyzes an uploaded image, returns a UI preview and detected corners.
- `POST /api/straighten`: Applies perspective warping based on user-supplied corners.
- `POST /api/enhance`: Applies the enhancement pipeline to output a clean binary image.
