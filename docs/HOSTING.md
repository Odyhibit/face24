# Hosting the demo online

The app is a standard Gradio app (`python app.py`). It needs about 1–2 GB of RAM (PyTorch + MediaPipe) and runs fine on a CPU, because each model has only ~4M parameters. Every option below needs `GRADIO_SERVER_NAME=0.0.0.0` so the server listens beyond localhost. Hugging Face sets this automatically.

> **Privacy:** when hosted, visitors' photos are uploaded to the server for processing. Generated QR/AVIF downloads are written to `output/` on the server. Say so on the page if you host it publicly.

## Option A — Hugging Face Space on ZeroGPU (free)

Free personal Hugging Face accounts can host up to 2 Gradio Spaces on ZeroGPU hardware. (Plain CPU Gradio Spaces need a PRO plan.) The app doesn't need a GPU, but ZeroGPU is the free Gradio tier. Free Spaces sleep when unused and wake on the next visit (expect a slow first load).

1. Create a Space at https://huggingface.co/new-space: SDK **Gradio**, hardware **ZeroGPU**.
2. Add this front matter to the top of the Space's `README.md`:

   ```yaml
   ---
   title: Face24 QR Codec
   emoji: 🙂
   colorFrom: gray
   colorTo: blue
   sdk: gradio
   sdk_version: 5.50.0
   python_version: "3.12"
   app_file: app.py
   pinned: false
   license: mit
   ---
   ```

3. Add a `packages.txt` containing `libgl1`. MediaPipe installs the full OpenCV build, which needs it on Linux.
4. Upload the repo contents. Files over 10 MB, including the checkpoints, are handled automatically by `huggingface_hub`:

   ```bash
   pip install huggingface_hub
   hf auth login
   hf upload <your-username>/face24-qr . . --repo-type=space --exclude ".git/*" "output/*" ".venv/*"
   ```

If ZeroGPU complains that no function uses the GPU, either decorate `encode_upload` in `demo/codec.py` with `@spaces.GPU` (and add `spaces` to `requirements.txt`) or use option B.

## Option B — Hugging Face Space on CPU Basic (PRO, $9/month)

Same steps as option A, but choose **CPU Basic** hardware. This is the least-friction setup if you're happy to pay for PRO.

## Option C — Google Cloud Run (pay-per-use, usually ~$0)

Package the app in a Docker image (`python:3.12-slim` + `libgl1` + `pip install -r requirements.txt`, `CMD python app.py`). Deploy with ≥2 GiB memory and `GRADIO_SERVER_NAME=0.0.0.0`, `GRADIO_SERVER_PORT=8080`. Cloud Run scales to zero, so a low-traffic demo usually stays within the free tier. It does need a billing account, and cold starts take a while.
