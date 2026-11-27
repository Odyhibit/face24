"""Presentation and event wiring for the local codec comparison."""
from pathlib import Path

import gradio as gr
from PIL import Image

EMPTY_AVIF_NOTE = "Encode a photograph to see the AVIF frame size chosen for each codec."
EMPTY_AVIF_CAPTION = "**AVIF** · —"


def original_heading(size=None, dimensions=None):
    title = f"{size:,} bytes" if size is not None else "— bytes"
    detail = f"{dimensions[0]:,} × {dimensions[1]:,} pixels · complete uploaded file" if dimensions else "Complete uploaded JPG or PNG file"
    return (f'<div class="card-heading"><span>ORIGINAL UPLOAD</span><h3>{title}</h3>'
            f'<p class="card-meta">{detail}</p></div>')


def build_app(runtime, aligner):
    from demo.codec import DEFAULT_PAYLOAD_BYTES, MODEL_CHOICES, decode_upload, encode_upload, preview_qr

    def cleared(message):
        # 8 images (original, aligned, 3 Face24/AVIF pairs) + QR + QR download, then the texts.
        return (*([None] * 10), message, EMPTY_AVIF_NOTE, *([EMPTY_AVIF_CAPTION] * 3), {}, original_heading())

    def encode_callback(path, payload_bytes):
        try:
            if path is None:
                raise ValueError("Choose a face photograph, then click Compare codecs.")
            path = Path(path)
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError("Choose a JPG or PNG photograph.")
            with Image.open(path) as image:
                image.load()
                values = encode_upload(image, aligner, runtime, payload_bytes)
                return (*values, original_heading(path.stat().st_size, image.size))
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            gr.Warning(str(exc))
            return cleared(str(exc))

    def choose_preview(payloads, payload_bytes):
        try:
            qr, download, status = preview_qr(payloads, payload_bytes, runtime)
            return qr, download, status, payload_bytes
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            gr.Warning(str(exc))
            return None, None, str(exc), gr.skip()

    def decode_callback(path):
        try:
            if path is None:
                raise ValueError("Choose a QR image, then click Decode QR.")
            with Image.open(path) as image:
                image.load()
                return decode_upload(image, runtime)
        except (OSError, TypeError, ValueError, RuntimeError) as exc:
            raise gr.Error(str(exc)) from exc

    theme = gr.themes.Base(
        primary_hue="teal", secondary_hue="slate", neutral_hue="slate",
        font=["Segoe UI", "Arial", "sans-serif"], font_mono=["Consolas", "monospace"],
    ).set(
        body_background_fill="#f4f6f8", body_background_fill_dark="#f4f6f8",
        body_text_color="#172a3a", body_text_color_dark="#172a3a",
        block_background_fill="#ffffff", block_background_fill_dark="#ffffff",
        block_border_color="#dce3e9", block_border_color_dark="#dce3e9",
        block_label_text_color="#475569", block_label_text_color_dark="#475569",
        input_background_fill="#ffffff", input_background_fill_dark="#ffffff",
        input_border_color="#cbd5e1", input_border_color_dark="#cbd5e1",
        button_primary_background_fill="#0f766e", button_primary_background_fill_dark="#0f766e",
        button_primary_text_color="#ffffff", button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#f1f5f9", button_secondary_background_fill_dark="#f1f5f9",
        button_secondary_text_color="#172a3a", button_secondary_text_color_dark="#172a3a",
    )
    # Keep every component legible with the same presentation when the browser
    # prefers a dark scheme; otherwise Gradio mixes dark labels into light cards.
    settings = theme.to_dict()["theme"]
    theme = theme.set(**{key: settings[key[:-5]] for key in settings
                         if key.endswith("_dark") and key[:-5] in settings})
    css = Path(__file__).with_name("styles.css").read_text(encoding="utf-8")
    with gr.Blocks(title="Face24 · Codec comparison", theme=theme, css=css, analytics_enabled=False) as demo:
        gr.HTML('''<header class="masthead"><div class="brand"><span class="brand-mark">F24</span>
          <span>FACE24 <span class="brand-sub">/ CODEC LAB</span></span></div>
          <span class="local-badge"><span></span> PROCESSED LOCALLY</span></header>
          <section class="hero"><p class="eyebrow">IMAGE COMPRESSION, SIDE BY SIDE</p>
          <h1>A face. A few hundred bytes.</h1>
          <p class="hero-copy">Compare three learned codecs with conventional AVIF compression at the same byte budget.
          One photograph, one aligned crop, three ways to carry it in a QR code.</p>
          <div class="workflow"><span><b>01</b> Add a photograph</span><span><b>02</b> Compare detail</span>
          <span><b>03</b> Export a QR</span></div></section>''')
        with gr.Tabs():
            with gr.Tab("Compare & encode"):
                payload_state = gr.State({})
                gr.Markdown("## 01 · Start with a photograph\nChoose a clear JPG or PNG with one visible face. Every codec, and AVIF, receives the same aligned 64 × 80 grayscale crop.", elem_classes="section-heading")
                with gr.Row(equal_height=True, elem_classes="source-row"):
                    with gr.Column(scale=2, min_width=260):
                        upload = gr.File(label="Face photograph", file_types=[".jpg", ".jpeg", ".png"], type="filepath", height=180)
                        encode_button = gr.Button("Compare codecs", variant="primary", size="lg")
                    with gr.Column(min_width=210, elem_classes="codec-card"):
                        original_title = gr.HTML(original_heading())
                        original = gr.Image(label="Original uploaded photograph", show_label=False, type="pil", format="png", height=250, interactive=False)
                        gr.Markdown("**File size** · Includes the original file’s compression and headers.", elem_classes="card-caption")
                    with gr.Column(min_width=210, elem_classes="codec-card"):
                        gr.HTML('<div class="card-heading"><span>SHARED INPUT · GRAYSCALE</span><h3>5,120 bytes</h3><p class="card-meta">64 × 80 pixels · 8 bits per pixel</p></div>')
                        aligned = gr.Image(label="Shared uncompressed input", show_label=False, type="pil", format="png", height=250, interactive=False, elem_id="face24-target")
                        gr.Markdown("**Raw pixels** · 64 × 80 × 1 byte. Excludes file headers; the preview PNG’s size is not counted.", elem_classes="card-caption")
                gr.Markdown("**Reading the sizes:** Original = uploaded file bytes; shared input = uncompressed grayscale pixels; reconstructions = bytes stored inside the QR. The original may have different dimensions and color channels.", elem_classes="size-context")
                gr.Markdown("## 02 · Compare the reconstructions\nEach Face24 reconstruction sits next to the smallest AVIF frame with at least as many bytes. Look at the eyes, mouth, and facial structure. Click a Face24 preview or its button to export that exact reconstruction’s QR.", elem_classes="section-heading")
                previews = []
                buttons = []
                avif_previews = []
                avif_captions = []
                with gr.Row(equal_height=True, elem_id="codec-cards"):
                    # Largest first, so the cards read 256 → 128 → 64 bytes left to right.
                    for title, subtitle, identifier, key in (
                        ("256 bytes", "LEARNED RVQ · PHASE 22", "phase22", 256),
                        ("128 bytes", "LEARNED · PHASE 21", "phase21", 128),
                        ("64 bytes", "LEARNED · PHASE 18", "phase18", 64),
                    ):
                        with gr.Column(min_width=300, elem_classes="codec-card"):
                            gr.HTML(f'<div class="card-heading"><span>{subtitle}</span><h3>{title}</h3></div>')
                            with gr.Row(elem_classes="pair-row"):
                                with gr.Column(min_width=120):
                                    preview = gr.Image(label=f"Face24 {title} reconstruction", show_label=False, type="pil", format="png", height=200, interactive=False, elem_id=f"face24-{identifier}")
                                    gr.Markdown(f"**Face24 · {key} bytes**", elem_classes="pair-caption")
                                with gr.Column(min_width=120):
                                    avif_preview = gr.Image(label=f"AVIF at least {title}", show_label=False, type="pil", format="png", height=200, interactive=False, elem_id=f"face24-avif{key}")
                                    avif_caption = gr.Markdown(EMPTY_AVIF_CAPTION, elem_classes="pair-caption")
                            button = gr.Button("Show QR", size="sm")
                            previews.append(preview)
                            avif_previews.append(avif_preview)
                            avif_captions.append(avif_caption)
                            buttons.append((button, key))
                with gr.Group(elem_classes="accounting-panel"):
                    gr.Markdown("### Every transmitted byte counts")
                    avif_note = gr.Markdown(EMPTY_AVIF_NOTE)
                    gr.Markdown("Each Face24 result is shown next to the smallest AVIF image that is at least as many bytes. Only the AVIF’s **compressed frame data** is counted: the rest of the file (about 280 bytes of container and AV1 headers) is identical for every face, so the decoder keeps it as a template, the same way the Face24 decoder keeps its codebook. Shared decoder software and model weights are excluded for both. AVIF is shown for comparison only; QR export covers the three Face24 codecs.")
                gr.Markdown("## 03 · Take the result with you\nSelect a codec or click any reconstruction above. The QR and download always use that preview’s original encoded bytes.", elem_classes="section-heading")
                with gr.Row(elem_classes="export-row"):
                    qr = gr.Image(label="Verified binary QR", type="pil", format="png", height=340, interactive=False, scale=1, min_width=280)
                    with gr.Column(scale=2, min_width=280):
                        choice = gr.Radio(choices=MODEL_CHOICES, value=DEFAULT_PAYLOAD_BYTES, label="Codec for QR export", elem_id="codec-choice")
                        status = gr.Textbox(label="Export status", value="Add a photograph to begin.", interactive=False, lines=2)
                        qr_download = gr.DownloadButton(label="Download QR as PNG", variant="primary")
                        gr.Markdown("**To decode:** save the QR, open the **Decode a QR** tab, and upload it. The format is detected automatically. Binary image payloads may not appear as text or a link in a phone’s camera app.")
                # encode_upload returns its images and captions in 64, 128, 256 order; map them
                # to the cards by size, not by the cards' on-screen order.
                by_size = {key: (preview, avif_preview, caption)
                           for (_, key), preview, avif_preview, caption in zip(buttons, previews, avif_previews, avif_captions)}
                pairs = [component for size in (64, 128, 256) for component in by_size[size][:2]]
                captions = [by_size[size][2] for size in (64, 128, 256)]
                outputs = [original, aligned, *pairs, qr, qr_download, status, avif_note, *captions, payload_state, original_title]
                qr_outputs = [qr, qr_download, status, choice]
                encode_button.click(encode_callback, inputs=[upload, choice], outputs=outputs, api_name="encode_callback", concurrency_id="face24_encoder", concurrency_limit=1)
                for preview, (button, key) in zip(previews, buttons):
                    def click_preview(payloads, payload_bytes=key):
                        return choose_preview(payloads, payload_bytes)
                    preview.select(click_preview, inputs=[payload_state], outputs=qr_outputs, api_name=f"preview_{key}_qr", scroll_to_output=True, concurrency_id="face24_encoder", concurrency_limit=1)
                    button.click(click_preview, inputs=[payload_state], outputs=qr_outputs, api_name=f"show_{key}_qr", scroll_to_output=True, concurrency_id="face24_encoder", concurrency_limit=1)
                choice.input(choose_preview, inputs=[payload_state, choice], outputs=qr_outputs, api_name="select_model_qr", concurrency_id="face24_encoder", concurrency_limit=1)
                upload.change(lambda: cleared("Click Compare codecs to encode this photograph."), outputs=outputs, api_name="clear_previews", concurrency_id="face24_encoder", concurrency_limit=1)
            with gr.Tab("Decode a QR"):
                gr.Markdown("## Bring a face back from its QR\nUpload a QR image exported by this demo or the browser version. It automatically recognizes the 64-, 128-, and 256-byte learned formats.", elem_classes="section-heading")
                with gr.Row():
                    with gr.Column():
                        qr_upload = gr.File(label="QR image · PNG or JPG", file_types=[".png", ".jpg", ".jpeg"], type="filepath")
                        decode_button = gr.Button("Decode QR", variant="primary")
                        decode_status = gr.Textbox(label="Detected format & payload size", interactive=False)
                    with gr.Column():
                        decoded = gr.Image(label="Recovered face · 64 × 80", type="pil", format="png", height=340, elem_id="face24-decoded", interactive=False)
                        decoded_download = gr.DownloadButton(label="Download recovered face")
                decode_button.click(decode_callback, inputs=qr_upload, outputs=[decoded, decoded_download, decode_status], api_name="decode_callback")
        gr.HTML('<footer class="page-footer"><span>FACE24 / CODEC LAB</span><span>Local processing · No image enhancement · Visual comparison, not an identity test</span></footer>')
    return demo
