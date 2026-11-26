"""Presentation and event wiring for the local codec comparison."""
from pathlib import Path

import gradio as gr
from PIL import Image

EMPTY_AVIF_NOTE = "Encode a photograph to see the AVIF file’s exact image-data and container-overhead sizes."


def avif_heading(size=None):
    title = f"{size} bytes" if size is not None else "≤ 512 bytes"
    return f'<div class="card-heading"><span>CONVENTIONAL · AVIF</span><h3>{title}</h3></div>'


def original_heading(size=None, dimensions=None):
    title = f"{size:,} bytes" if size is not None else "— bytes"
    detail = f"{dimensions[0]:,} × {dimensions[1]:,} pixels · complete uploaded file" if dimensions else "Complete uploaded JPG or PNG file"
    return (f'<div class="card-heading"><span>ORIGINAL UPLOAD</span><h3>{title}</h3>'
            f'<p class="card-meta">{detail}</p></div>')


def build_app(runtime, aligner):
    from demo.codec import DEFAULT_PAYLOAD_BYTES, MODEL_CHOICES, decode_upload, encode_upload, preview_qr

    def cleared(message):
        return (*([None] * 8), message, EMPTY_AVIF_NOTE, None, {}, avif_heading(), original_heading())

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
                return (*values, avif_heading(len(values[-1]["avif"])),
                        original_heading(path.stat().st_size, image.size))
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
          <p class="hero-copy">Compare three learned codecs with conventional AVIF compression.
          One photograph, one aligned crop, four ways to carry it in a QR code.</p>
          <div class="workflow"><span><b>01</b> Add a photograph</span><span><b>02</b> Compare detail</span>
          <span><b>03</b> Export a QR</span></div></section>''')
        with gr.Tabs():
            with gr.Tab("Compare & encode"):
                payload_state = gr.State({})
                gr.Markdown("## 01 · Start with a photograph\nChoose a clear JPG or PNG with one visible face. All four codecs receive the same aligned 64 × 80 grayscale crop.", elem_classes="section-heading")
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
                gr.Markdown("## 02 · Compare the reconstructions\nLook at the eyes, mouth, and facial structure. Click a preview or its button to export that exact reconstruction’s QR.", elem_classes="section-heading")
                previews = []
                buttons = []
                with gr.Row(equal_height=True, elem_id="codec-cards"):
                    for title, subtitle, identifier, key in (
                        ("64 bytes", "LEARNED · PHASE 18", "phase18", 64),
                        ("128 bytes", "LEARNED · PHASE 21", "phase21", 128),
                        ("256 bytes", "LEARNED RVQ · PHASE 22", "phase22", 256),
                        ("≤ 512 bytes", "CONVENTIONAL · AVIF", "avif", "avif"),
                    ):
                        with gr.Column(min_width=210, elem_classes="codec-card"):
                            heading = gr.HTML(f'<div class="card-heading"><span>{subtitle}</span><h3>{title}</h3></div>')
                            if key == "avif":
                                avif_title = heading
                            preview = gr.Image(label=f"{title} reconstruction", show_label=False, type="pil", format="png", height=250, interactive=False, elem_id=f"face24-{identifier}")
                            button = gr.Button("Show QR", size="sm")
                            previews.append(preview)
                            buttons.append((button, key))
                with gr.Group(elem_classes="accounting-panel"):
                    gr.Markdown("### Every transmitted byte counts")
                    avif_note = gr.Markdown(EMPTY_AVIF_NOTE)
                    gr.Markdown("The learned sizes count raw indices and rely on pre-shared neural decoders. AVIF’s size counts the **complete standalone file**, including its container, and relies on a standard AVIF decoder. Shared decoder software and model weights are excluded from both. QR framing and error correction add transport overhead for every option; the displayed sizes count the bytes stored inside the QR.")
                    avif_download = gr.DownloadButton(label="Download complete AVIF file", size="sm")
                gr.Markdown("## 03 · Take the result with you\nSelect a codec or click any reconstruction above. The QR and download always use that preview’s original encoded bytes.", elem_classes="section-heading")
                with gr.Row(elem_classes="export-row"):
                    qr = gr.Image(label="Verified binary QR", type="pil", format="png", height=340, interactive=False, scale=1, min_width=280)
                    with gr.Column(scale=2, min_width=280):
                        choice = gr.Radio(choices=MODEL_CHOICES, value=DEFAULT_PAYLOAD_BYTES, label="Codec for QR export", elem_id="codec-choice")
                        status = gr.Textbox(label="Export status", value="Add a photograph to begin.", interactive=False, lines=2)
                        qr_download = gr.DownloadButton(label="Download QR as PNG", variant="primary")
                        gr.Markdown("**To decode:** save the QR, open the **Decode a QR** tab, and upload it. The format is detected automatically. Binary image payloads may not appear as text or a link in a phone’s camera app.")
                outputs = [original, aligned, *previews, qr, qr_download, status, avif_note, avif_download, payload_state, avif_title, original_title]
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
                gr.Markdown("## Bring a face back from its QR\nUpload a QR image exported by this demo. It automatically recognizes the 64-, 128-, and 256-byte learned formats and complete AVIF files up to 512 bytes.", elem_classes="section-heading")
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
