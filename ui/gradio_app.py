import os
from pathlib import Path
import gradio as gr
from app.cache import Cache
from app.orchestrator import Orchestrator


def _build_orch():
    stub_stages = {
        int(s) for s in os.getenv("DREAMSCAPE_STUB_STAGES", "3,4,6").split(",") if s.strip()
    }
    cache_dir = Path(os.getenv("DREAMSCAPE_CACHE_DIR", "cache"))
    cache = Cache(db_path=cache_dir / "runs.db", asset_dir=cache_dir / "assets")
    return Orchestrator(cache=cache, stub_stages=stub_stages)


def build_ui() -> gr.Blocks:
    from ui.rater_study import build_rater_tab

    orch = _build_orch()

    with gr.Blocks(title="DreamScapeAI") as demo:
        gr.Markdown("# DreamScapeAI\nTransform a text prompt into a cinematic short video.")

        with gr.Tabs():
            with gr.TabItem("Generate"):
                with gr.Row():
                    with gr.Column(scale=2):
                        prompt_box = gr.Textbox(
                            label="Story Prompt",
                            placeholder="A lone wolf howls at a full moon in a snowy mountain.",
                            max_lines=4,
                        )
                        with gr.Row():
                            duration_radio = gr.Radio(choices=[30, 60, 90], value=60, label="Duration (s)")
                            style_dd = gr.Dropdown(
                                choices=["cinematic", "documentary", "anime", "noir", "horror"],
                                value="cinematic", label="Style",
                            )
                            voice_radio = gr.Radio(choices=["female", "male"], value="female", label="Voice")
                        generate_btn = gr.Button("Generate Video", variant="primary")
                    with gr.Column(scale=1):
                        status_box = gr.Textbox(label="Status", interactive=False)

                with gr.Tabs():
                    with gr.TabItem("Scene Plan"):
                        scene_table = gr.Dataframe(
                            headers=["Scene", "Description", "Narration", "Mood", "Duration (s)"],
                            interactive=False,
                        )
                    with gr.TabItem("Images"):
                        image_gallery = gr.Gallery(label="Generated Images", columns=4)
                    with gr.TabItem("Output"):
                        video_out = gr.Video(label="Generated Video")
                        download_file = gr.File(label="Download MP4")

                run_state = gr.State({})

                def on_generate(prompt, duration, style, voice):
                    if not prompt.strip():
                        return "Please enter a prompt.", None, None, None, None, {}
                    try:
                        run = orch.run_pipeline(prompt.strip(), int(duration), style, voice)
                    except ValueError as e:
                        return str(e), None, None, None, None, {}

                    scenes = run.scene_plan.scenes if run.scene_plan else []
                    rows = [[s.id, s.description[:60], s.narration_text[:60], s.mood,
                             f"{s.duration_estimate_s:.1f}"] for s in scenes]
                    img_paths = [img.path for img in run.visual_output.images] if run.visual_output else []
                    video_path = run.video_output.path if run.video_output else None
                    status = (f"Done — run {run.run_id} | {run.video_output.duration_s:.1f}s"
                              if run.video_output else "Failed")
                    return status, rows, img_paths, video_path, video_path, {"run_id": run.run_id}

                generate_btn.click(
                    fn=on_generate,
                    inputs=[prompt_box, duration_radio, style_dd, voice_radio],
                    outputs=[status_box, scene_table, image_gallery, video_out, download_file, run_state],
                )

            with gr.TabItem("Rate videos"):
                build_rater_tab()

    return demo


if __name__ == "__main__":
    build_ui().launch()
