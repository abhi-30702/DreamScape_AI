# DreamScapeAI — Product Requirements Document (PRD)

**Project:** DreamScapeAI — AI Story-to-Video Generation System
**Type:** Final Year Research Project
**Domain:** Generative AI · NLP · Computer Vision · Multimodal Synthesis
**Version:** 1.1 (Revised)
**Status:** Draft / Pre-Development
**Last Updated:** June 2026

---

## 1. Executive Summary

DreamScapeAI is an end-to-end multimodal AI system that transforms a single natural-language prompt into a complete cinematic short video. The output includes synchronized visuals, AI-generated narration, automatic subtitles, and emotion-aware background music — all assembled into a downloadable MP4.

The project's core research contribution is a **multimodal synchronization engine**: a coordination layer that aligns four independently-generated AI modalities (text, image, speech, music) into a temporally and emotionally coherent narrative. Today's AI tools operate in silos; DreamScapeAI unifies them through an explicit orchestration protocol.

**Key Constraints (v1.1):**
- Single narrator (no dialogue / multi-speaker support)
- No character consistency across scenes (each scene independently generated)
- Sequential pipeline execution (not parallel) with aggressive caching
- Fully local models (zero API costs; fully reproducible)
- Target generation time: **10 minutes per 60-second video** on T4 GPU

---

## 2. Problem Statement

Current generative AI tools are fragmented:

- Text generators (GPT, Llama) produce stories but no visuals
- Image models (Stable Diffusion, DALL·E) produce images but no narrative continuity
- Speech engines (XTTS, Coqui) produce voice but no scene context
- Music generators (MusicGen, Suno) produce audio but no emotional grounding

A user who wants a complete short video must manually orchestrate 4–6 separate tools, then edit everything together in Adobe Premiere or Final Cut Pro. There is no unified pipeline that takes a prompt and returns a finished cinematic video with all modalities aligned. DreamScapeAI closes this gap.

**Gap in existing solutions:**
- Make-A-Video and Stable Video Diffusion generate video directly, but lack integrated narration/music
- VideoPoet is closed-source and requires compute beyond academic budgets
- OpenAI Sora is not accessible to researchers
- No published system demonstrates **measurable temporal/emotional alignment** across all four modalities

---

## 3. Goals & Objectives

| # | Objective | Success Criterion |
|---|---|---|
| 1 | End-to-End Generation | Produce a 30–90 second video from a single prompt |
| 2 | Full Synchronization | Audio-visual sync error <200 ms; music-mood coherence rated ≥3.5/5 in study |
| 3 | Interactive Demo | Web UI deployed publicly (Hugging Face Spaces) with <10 min generation time |
| 4 | Research Contribution | Novel synchronization architecture + emotion propagation protocol documented in thesis |
| 5 | Human Evaluation | Structured user study with ≥20 participants (pilot scale, realistic for 20-week timeline) |
| 6 | Reproducibility | All code, prompts, evaluation rubrics published on GitHub; models on HF Hub |

### Non-Goals (v1)
- Long-form video (>2 minutes) — deferred to v2
- Real-time generation (<10 seconds end-to-end) — deferred to v2 with hardware optimization
- Character consistency across scenes — explicitly deferred to v2 (IP-Adapter + seed control)
- Lip-sync animation — deferred to v2 (separate image-to-video stage)
- Multiple speakers / dialogue — deferred to v2 (voice cloning + speaker identification)
- Per-scene music generation — v1 uses single background track for performance

---

## 4. Target Users

- **Primary:** Researchers and academics evaluating multimodal AI coherence (thesis defense, conference papers)
- **Secondary:** Content creators (educators, indie storytellers, social media creators) who need rapid concept videos for prototyping
- **Tertiary:** Students and practitioners learning generative AI pipeline design

---

## 5. User Stories

- *As a researcher,* I enter "A lonely astronaut discovers a glowing forest on Mars" and receive a 60-second video with narration, visuals, subtitles, and music within 10 minutes. I can inspect intermediate outputs (scene breakdown, image prompts, audio waveforms) to evaluate synchronization quality.

- *As a content creator,* I use DreamScapeAI to rapidly prototype a concept video without hiring a video editor, then refine it in professional software.

- *As an evaluator,* I can rate the generated video on: visual quality, narration clarity, music-mood fit, audio-visual sync, and overall narrative coherence on 5-point Likert scales.

- *As a researcher,* I can compare this integrated pipeline (DreamScapeAI) against an unintegrated baseline (running each tool separately and manually combining them) to quantify the value of the orchestration layer.

- *As a developer,* I can swap any model in the pipeline (e.g., replace Llama-3.1-8B with GPT-4o, or Stable Diffusion with FLUX) without breaking the system, as long as the input/output contracts are respected.

---

## 6. System Architecture

### 6.1 Pipeline Stages

```
[User Prompt]
        ↓
[Stage 1: Prompt Parsing]        (LLM)
        ↓
[Stage 2: Scene Expansion]        (LLM)
        ↓
[Stage 3: Visual Generation]      (SDXL)
        ↓
[Stage 4: Narration Generation]   (XTTS-v2)
        ↓
[Stage 5: Subtitle Generation]    (Whisper)
        ↓
[Stage 6: Music Generation]       (MusicGen)
        ↓
[Stage 7: Video Assembly]         (MoviePy + FFmpeg)
        ↓
[MP4 Download]
```

**Execution Model:** Stages run **sequentially**. Intermediate outputs are cached (24-hour TTL). If a user regenerates a single stage (e.g., re-narrate Scene 3), downstream stages are automatically re-executed, but upstream stages are skipped.

---

### 6.2 Stage Detail

| Stage | Input | Output | Model | Compute | Notes |
|---|---|---|---|---|---|
| 1. Prompt Parsing | User text (≤500 chars) | Structured intent, sentiment (sad/neutral/happy), duration target | Llama-3.1-8B (Ollama) | 2–4 GB VRAM | Extracts: story arc, tone, key entities |
| 2. Scene Expansion | Intent + sentiment | JSON: 4–8 scenes with descriptions (100 tokens each), durations, mood tags | Llama-3.1-8B (Ollama) | 2–4 GB VRAM | Distributes 60s across scenes (e.g., 10s + 15s + 12s + 8s + 15s) |
| 3. Visual Generation | Scene description + style | 1 image per scene (1024×576, 16:9) | Stable Diffusion XL (diffusers, local) | 10–12 GB VRAM | ~30–40s per image on T4 (50 steps). No seed variation; same prompt = same image. |
| 4. Narration | Scene text + sentiment | Per-scene audio WAV + timestamps (start/end in seconds) | XTTS-v2 (Coqui TTS, local) | 4–6 GB VRAM | ~10–20s per 30-second narration clip. Voice & emotion controlled via speaker embedding. |
| 5. Subtitle Generation | Narration WAV + source text | SRT file (timecode, text) | Whisper (small or base, local) | 2–3 GB VRAM | ~5–10s per audio clip. Uses Whisper's automatic timestamps (word-level alignment). |
| 6. Music Generation | Mood tags (csv: happy, sad, energetic) + target duration | 1 WAV track (matches total video duration) | MusicGen-medium (Meta, local) | 6–8 GB VRAM | ~60–90s per 90-second track. Conditioned on text description derived from mood tags. |
| 7. Video Assembly | Images + narration WAV + music WAV + SRT | MP4 (H.264, 1080p, 30 fps) | MoviePy + FFmpeg | 2–4 GB VRAM | Compose: image for N seconds, overlay audio, burn subtitles, letterbox music under narration (–18 dB). |

---

### 6.3 Orchestration Layer (Novel Research Contribution)

A central `Orchestrator` class (Python) coordinates all stages. **Key responsibilities:**

#### 6.3.1 Temporal Synchronization
- **TTS-First Timing:** Scene durations are NOT predefined. Instead:
  1. Expand scenes (Stage 2) with rough duration estimates
  2. Generate narration (Stage 4) for each scene
  3. Measure actual audio duration
  4. Adjust scene display duration to match narration (±2 second window)
  5. If aggregated duration drifts, re-distribute across scenes proportionally
- **Formula:** `scene_display_duration = narration_audio_duration + fade_in(0.5s) + fade_out(0.5s)`
- **Tolerance:** Total video ±2 seconds from user-requested target (e.g., user requests 60s, accept 58–62s)

#### 6.3.2 Emotion-Aware Propagation
The orchestrator extracts a single sentiment label from the user prompt (Stage 1) and propagates it across modalities:

| Sentiment | SDXL Negative Prompt | MusicGen Text Condition | XTTS Speaker Setting | Notes |
|---|---|---|---|---|
| Happy | "dark, bleak, sad, monochrome" | "uplifting, bright, major key, fast tempo" | Female speaker, pitch +2, speed 0.95 | Cheerful & energetic |
| Neutral | (empty) | "calm, ambient, minimal, mid-tempo" | Neutral male speaker, pitch 0, speed 1.0 | Objective narrator |
| Sad | "bright, colorful, cheerful, optimistic" | "melancholic, minor key, slow tempo, sparse" | Male speaker, pitch –1, speed 1.05 | Reflective & deliberate |

**Example Application:**
- User prompt: "A lost child finds home" (extracted sentiment: hopeful/bittersweet → maps to "happy" with undertones)
- SDXL negative prompt: "dark, bleak, hopeless"
- MusicGen condition: "uplifting yet tender, major key, moderate tempo"
- XTTS: Female voice, slightly higher pitch, warm delivery

#### 6.3.3 Caching & Regeneration
- **Asset Cache:** Outputs stored at `/tmp/dreamscape_cache/{prompt_hash}/` with 24-hour TTL
- **Selective Regeneration:** User can click "Regenerate Scene 3 Narration" without re-running Stages 1–3 (LLM, visual gen)
- **Downstream Auto-Run:** Regenerating Stage 4 (narration) automatically triggers Stage 5 (subtitles) and Stage 7 (assembly)
- **Cache Invalidation:** If user edits scene text, only that scene and downstream stages invalidate

#### 6.3.4 Failure Recovery
- **Graceful Degradation:** If SDXL generation fails (NSFW filter, OOM, timeout):
  1. Retry with simpler prompt (remove adjectives, lower resolution 512×384)
  2. If still fails, use a fallback placeholder image (solid color + text overlay)
  3. Log error and continue (video is produced, but with degraded visuals)
- **Timeout Management:** Each stage has a 5-minute timeout. If exceeded, fall back to simpler model or placeholder
- **OOM Handling:** VRAM monitoring. If GPU hits 95%, offload CPU-friendly stages to CPU

---

### 6.4 Data Flow: Detailed Example

**Input:** "A lone wolf howls at a full moon in a snowy mountain."

**Stage 1 Output:**
```json
{
  "prompt": "A lone wolf howls at a full moon in a snowy mountain.",
  "sentiment": "melancholic",
  "duration_target_s": 60,
  "style": "cinematic",
  "key_entities": ["wolf", "moon", "snow", "mountain"]
}
```

**Stage 2 Output:**
```json
{
  "scenes": [
    {
      "id": 0,
      "description": "Wide shot of a snow-covered mountain under starlight, wind rustling through pine trees.",
      "narration_text": "In the heart of winter, where silence reigns supreme...",
      "mood": "melancholic",
      "duration_estimate_s": 12
    },
    {
      "id": 1,
      "description": "Close-up of wolf's face, eyes glowing golden, ears perked toward the sky.",
      "narration_text": "A lone wolf emerges from the darkness, sensing a call from the heavens.",
      "mood": "melancholic",
      "duration_estimate_s": 15
    },
    {
      "id": 2,
      "description": "Full moon rising above the mountain peak, perfectly round and luminescent.",
      "narration_text": "The full moon breaks through the clouds, ancient and eternal.",
      "mood": "melancholic",
      "duration_estimate_s": 18
    },
    {
      "id": 3,
      "description": "Wolf silhouetted against the moonlit sky, head thrown back, mouth open in a haunting howl.",
      "narration_text": "And the wolf's cry echoes across the frozen landscape, a song of solitude and survival.",
      "mood": "melancholic",
      "duration_estimate_s": 15
    }
  ]
}
```

**Stage 3 Output:**
4 images (1024×576 each), SDXL-generated with negative prompt: "bright, colorful, cheerful, optimistic"

**Stage 4 Output:**
4 audio files:
- Scene 0: 11.2 seconds of narration (measured from TTS output)
- Scene 1: 14.8 seconds
- Scene 2: 17.5 seconds
- Scene 3: 15.3 seconds
- **Total: 58.8 seconds** (within 60 ± 2 target ✓)

**Stage 5 Output:**
```srt
1
00:00:00,000 --> 00:00:02,150
In the heart of winter, where silence reigns supreme...

2
00:00:12,200 --> 00:00:27,000
A lone wolf emerges from the darkness, sensing a call from the heavens.

... (continues)
```

**Stage 6 Output:**
One WAV file, 58.8 seconds, melancholic minor-key piano + ambient strings, slow tempo (80 BPM), "minor key, slow tempo, sparse, tender"

**Stage 7 Output:**
MP4 (1080p, H.264, 30 fps), 58.8 seconds:
- 0–11.2s: Image 0 displayed, narration plays, music at –6 dB
- 11.2–26.0s: Image 1 displayed, narration plays, music at –6 dB
- 26.0–43.5s: Image 2 displayed, narration plays, music at –6 dB
- 43.5–58.8s: Image 3 displayed, narration plays, music at –6 dB
- Subtitles burned at bottom, white text, 14pt Arial

---

## 7. Technical Stack

| Layer | Technology | Rationale | Constraints |
|---|---|---|---|
| **Language Model (Scene Expansion)** | Llama-3.1-8B via Ollama (local) | Free, deterministic, reproducible. No API costs. No rate limits. | Slightly lower coherence than GPT-4o; acceptable for scene planning. |
| **Image Generation** | Stable Diffusion XL (diffusers library, local) | Open-source, high quality, fully controllable via negative prompts. | ~30–40s per 1024×576 image on T4. Not real-time. |
| **Text-to-Speech** | XTTS-v2 (Coqui TTS, local) | Multilingual, expressive, open-source. Supports speaker control. | Slightly robotic vs. ElevenLabs; acceptable for narration. Can be swapped for ElevenLabs in paid tier. |
| **Music Generation** | MusicGen-medium (Meta, local) | Best open-source music model. Conditionable on text + duration. | ~60–90s per track. One track per video (not per-scene) in v1. |
| **Automatic Speech Recognition** | Whisper (OpenAI, local via whisper.cpp) | Best open ASR. Word-level timestamp accuracy. | Timestamps are ~100–200ms accurate; acceptable for subtitle sync. |
| **Video Composition** | MoviePy (Python) + FFmpeg | Native Python API. Handles image sequencing, audio mixing, subtitle burning. | CPU-bound for large videos; offload to CPU if GPU low on VRAM. |
| **Caching & State** | SQLite (local) + filesystem (asset storage) | Simple, no external dependencies. Fast for <1MB JSON state. | Growth: ~500 MB cache per 10 videos. 24-hour TTL cleanup. |
| **Frontend (MVP)** | Gradio (Python) | Fastest ML demo framework. Built-in progress bars, file upload/download. | Limited styling; adequate for research. React migration in v2 if needed. |
| **Frontend (Production)** | React + TypeScript + Vite | Modern, responsive, keyboard-accessible. Optional for v1. | Adds 2–3 weeks of UI engineering; recommend post-thesis. |
| **Backend** | FastAPI (Python) | Async, type-safe, native ML ecosystem integration. | Single-worker (HF Spaces constraint). Queue system deferred to v2. |
| **GPU Platform (Primary)** | Hugging Face Spaces (ZeroGPU) | Free T4 GPU tier. Persistent storage. Easy deployment. | ~5-minute timeout per execution; T4 is bottleneck. |
| **GPU Platform (Backup)** | Google Colab Pro / RunPod | Larger GPUs (A100, H100) for speed testing | Cost: $10–20 per hour for RunPod. Use for benchmarking, not production. |
| **Storage** | HF Hub (model weights) + Hugging Face Datasets (evaluation data) | Free tier sufficient for ~50 GB. Versioning built-in. | Models (SDXL, MusicGen) = 12–15 GB; host on HF Hub, not GitHub. |
| **Code Repository** | GitHub (public) | Linked from thesis, accessible to reviewers. | Document setup: "clone repo, download models from HF Hub, run via Gradio." |

---

## 8. Functional Requirements

### FR-1: Prompt Input
- Accept text input up to **500 characters**
- Optional: **duration target** (30s / 60s / 90s, default 60s)
- Optional: **style preset** (cinematic, documentary, anime, noir, horror) — influences SDXL negative prompts
- Optional: **voice selection** (female / male, language defaults to English) — passed to XTTS speaker ID
- Input validation: reject profanity, copyrighted character names, violence keywords
- Error messaging: clear feedback if prompt is rejected (e.g., "Prompts must not contain copyrighted character names")

### FR-2: Scene Decomposition
- Generate **4–8 scenes** from the prompt (distribution determined by LLM heuristics)
- Each scene contains:
  - Visual description (100–150 tokens)
  - Narration text (30–100 tokens)
  - Mood tag (happy / neutral / sad)
  - Duration estimate (seconds, overridden by actual TTS duration in Stage 4)
- UI: Display scene preview as a **scrollable plan** before generation begins
- Editability: User can modify scene text (description, narration) before proceeding (causes Stage 4+ to re-run)
- Confirmation: User must explicitly click "Generate Video" to proceed (prevents accidental long-running jobs)

### FR-3: Visual Generation
- Generate one 1024×576 image per scene (16:9 widescreen)
- Consistency across scenes via:
  - **Shared negative prompt** (derived from sentiment; see Section 6.3.2)
  - **Fixed seed** (optional, toggled per-scene): if toggled ON, image is reproducible; if OFF, new image on regeneration
  - **LoRA-free v1:** No IP-Adapter or character consistency in v1 (explicitly noted as v2 feature)
- Style injection: Apply selected style preset via negative prompt (e.g., "anime" style removes "photorealistic, detailed")
- Error handling: If NSFW filter triggers, auto-regenerate with simplified prompt (remove adjectives) up to 3 times; if still fails, use placeholder (solid color + "Filtered for safety")
- Display: Show generated image in preview; allow one-click regenerate per scene

### FR-4: Audio Generation (Narration & Music)
**Narration:**
- Generate per-scene narration matching scene text using XTTS-v2
- Voice selection: Passed to XTTS (speaker ID)
- Emotion modulation: Sentiment controls pitch/speed (Section 6.3.2)
- Output: Per-scene WAV files with **measured duration** (not estimated)
- **Stage 4 output drives Stage 3 duration:** Images display for exactly the narration duration (plus 1s fade-in/out)

**Music:**
- Generate **one background track** for the entire video (not per-scene)
- Duration: exactly match the aggregate video duration (sum of all scene narration + fades)
- Conditioning: Text description derived from mood tags (e.g., "melancholic, minor key, slow tempo, sparse, tender")
- Audio mixing: Music volume auto-adjusted to –6 dB during narration (ducking), –2 dB during silence
- Format: Mono or stereo WAV, 48 kHz

### FR-5: Subtitle Generation & Timing
- Auto-generate SRT subtitles from narration audio using Whisper
- Whisper processes each scene's audio separately; word-level timestamps are concatenated into SRT format
- Subtitle display: White Arial 14pt, center-bottom of frame, max 40 characters per line
- Timing tolerance: **Subtitle-to-audio sync error <200 ms** (measured via waveform cross-correlation in evaluation)
- Optional: User can toggle subtitles ON/OFF in final MP4 (burned into video, not separate SRT file for simplicity in v1)

### FR-6: Video Assembly
- Combine all assets into a single MP4:
  - Images (displayed for narration duration)
  - Narration audio (centered)
  - Music audio (ducked beneath narration)
  - Subtitles (burned in)
- Codec: H.264, 1080p (1920×1080 upscaled from 1024×576 source), 30 fps
- Aspect ratio: 16:9 (letterbox to match)
- Duration: Automatically match sum of all scene narrations
- File size: ~100–200 MB for 60-second video (typical)
- Output: Downloadable MP4 + inline HTML5 player in Gradio

### FR-7: Inspection & Regeneration Mode
- Display intermediate artifacts for debugging:
  - Scene plan (JSON-like preview)
  - Generated images (gallery view)
  - Narration audio waveforms (with Whisper transcript overlay)
  - Music waveform
- Regeneration UI:
  - "Regenerate Scene [N] Image" → re-runs Stage 3 only for that scene; downstream stages auto-trigger
  - "Regenerate Scene [N] Narration" → re-runs Stage 4 for that scene; triggers Stage 5 (subtitles) and Stage 7 (assembly)
  - "Regenerate Music" → re-runs Stage 6 only; triggers Stage 7 (assembly)
  - "Regenerate All" → full pipeline restart
- Timing: Regenerations should complete in <3 minutes (music slowest step)

### FR-8: Safety & Content Filtering
**Prompt Filtering:**
- Reject prompts containing:
  - Copyrighted character names (maintained as blocklist: "Mickey Mouse", "Darth Vader", etc.)
  - Excessive violence keywords: "kill", "murder", "gore", "weapon"
  - Explicit sexual content keywords
- Use simple keyword matching (v1), upgrade to BERT classifier in v2
- User-facing error: "Your prompt contains restricted content. Please revise and try again."

**Image Filtering:**
- Use Stability AI's built-in NSFW safety checker (enabled by default in diffusers library)
- If image is flagged, auto-regenerate with simplified prompt (remove adjectives, lower contrast)
- If failed after 3 retries, replace with placeholder (solid color + "Image filtered for safety")

---

## 9. Non-Functional Requirements

| Category | Requirement | Rationale | Measurement |
|---|---|---|---|
| **Performance** | End-to-end generation ≤10 minutes for 60s video on T4 GPU | Realistic for SDXL (30–40s/image × 6 scenes = 3–4 min) + XTTS (2–3 min) + MusicGen (1.5 min) | Track timing per stage; log in results |
| **Quality (Coherence)** | Mean Opinion Score (MOS) ≥3.5/5 on narrative coherence in user study | Passing grade for research contribution | 20+ raters, 10 videos each, 5-point Likert scale |
| **Quality (Sync)** | Audio-visual sync error <200 ms measured via waveform cross-correlation | Within human perception threshold | Automated metric + manual spot-check |
| **Reliability** | Pipeline success rate ≥85% on a 100-prompt benchmark | At least 85 of 100 prompts produce valid MP4 | Log all failures by stage; categorize (NSFW, OOM, timeout) |
| **Safety** | Zero high-severity NSFW/violence outputs in evaluation dataset | Legal/ethical requirement | Manual review of all 10 user study videos |
| **Cost (Free Tier)** | <$0.01 per video using fully local models (Llama, XTTS, MusicGen, SDXL) | Reproducible research | Serverless on HF Spaces (Compute = $0, models = download once) |
| **Accessibility** | Keyboard navigation in UI; subtitles on by default; alt text for images | WCAG 2.1 AA compliance | Audit before thesis defense |
| **Reproducibility** | Same prompt + seed → identical output (deterministic) | Scientific rigor | Test with 10 prompts; hash outputs |
| **Storage** | Cache <5 GB for 100 videos (48-hour retention) | Cost & performance | Monitor `/tmp/dreamscape_cache/` size |

---

## 10. Research Contributions

This is a **research project**, not just a product. The thesis must defend these **four novel contributions:**

### 10.1 Multimodal Synchronization Engine
**Definition:** A formal coordination protocol that maintains temporal and emotional alignment across four independently-trained generative models.

**Novelty:** Existing pipelines (Make-A-Video, VideoPoet) generate video directly, losing narration/music synchronization. DreamScapeAI generates each modality separately, then synchronizes via:
1. **TTS-first timing:** Narration duration drives image display duration
2. **Explicit temporal contract:** Each stage outputs timestamps; Orchestrator validates alignment
3. **Quantifiable sync error:** Measured via waveform cross-correlation; target <200 ms

**Evaluation in thesis:**
- Define formal "synchronization error" metric
- Benchmark DreamScapeAI vs. manual pipeline (all tools run separately, combined by hand)
- Show <10% improvement over manual is not acceptable; target >30% improvement

### 10.2 Emotion-Aware Storytelling
**Definition:** An affective computing pipeline that extracts a single sentiment label from the user prompt and propagates it across all downstream modalities (visuals, speech, music).

**Novelty:** Sentiment propagation is underexplored in multimodal generation. Existing systems handle text emotion (NLP) or audio emotion (speech synthesis), but not **coordinated cross-modal emotion**.

**Concrete Example:**
- Prompt: "A lost child finds home" → extracted sentiment: "hopeful"
- SDXL: Use negative prompt "dark, bleak, hopeless" to push toward warm imagery
- XTTS: Female voice, pitch +2 semitones, speed 0.95 (slightly faster, more energetic)
- MusicGen: "uplifting yet tender, major key, moderate tempo"

**Evaluation in thesis:**
- Conduct ablation study: generate 5 videos WITH emotion propagation, 5 WITHOUT (neutral defaults)
- Show 20+ raters: "Rate emotional coherence (1=incoherent, 5=highly coherent)"
- Hypothesis: Videos WITH emotion propagation score >1 point higher on average

### 10.3 Adaptive Scene Planning via LLM
**Definition:** An LLM-driven scene decomposition method that dynamically distributes a target video duration across scenes, balancing narrative pacing with technical constraints.

**Novelty:** Scene planning typically requires manual storyboarding. DreamScapeAI automates this:
1. User specifies target duration (60s)
2. LLM generates scene structure (4–8 scenes) with duration estimates
3. Stage 4 generates actual narration, measuring real duration
4. Orchestrator re-distributes if aggregate drifts >2s from target

**Evaluation in thesis:**
- Log duration distribution for 50 prompts
- Measure: % prompts where final duration within ±2s of target
- Measure: LLM-estimated duration vs. actual TTS duration (absolute error ≤5s?)

### 10.4 Human-Centered Evaluation Framework for Multimodal Coherence
**Definition:** A reproducible protocol for measuring narrative coherence across four modalities via structured human ratings.

**Novelty:** Most multimodal papers use automatic metrics (CLIP, BLEU). Few use human evaluation for **narrative coherence** specifically. This framework provides:
1. **Evaluation rubric:** Defines "coherence" (visual-narrative match, music-mood match, audio-visual sync, pacing)
2. **Rater instructions:** Clear definitions so raters agree (Fleiss' kappa ≥0.6 target)
3. **Comparison baseline:** Same 10 prompts, manually combined (no orchestration)
4. **Statistical analysis:** Report mean/SD/CI per dimension; paired t-tests

**Deliverables in thesis:**
- Evaluation rubric (PDF, reproducible)
- Rater instructions + training slides
- Data (anonymized ratings from 20+ raters × 10 videos = 200 ratings)
- Fleiss' kappa inter-rater reliability
- Mean opinion scores per dimension

---

## 11. Evaluation Plan

### 11.1 Automated Metrics (per video)

| Metric | Definition | Target | Tool |
|---|---|---|---|
| **CLIPScore (per scene)** | Image-text alignment via CLIP embedding cosine similarity | ≥0.70 | clip-score library |
| **Whisper WER** | Word Error Rate: transcribe generated narration, compare to source text | ≤10% | Whisper (transcribe → compare) |
| **Sync Error** | Time offset between subtitle timing and actual narration audio; measured via waveform cross-correlation | <200 ms | scipy.signal.correlate + manual timestamps |
| **Pipeline Success Rate** | % of prompts producing valid MP4 without human intervention | ≥85% | Logging per stage |
| **Duration Accuracy** | Final video duration vs. user target | ±2 seconds | ffprobe to measure |

### 11.2 Human Evaluation (20+ participants, pilot scale)

**Study Design:**
- 20 participants (students + researchers)
- Each rates **10 videos** (5 generated by DreamScapeAI, 5 from manual baseline)
- Rating scale: 5-point Likert (1=Strongly Disagree, 5=Strongly Agree)
- Duration: ~15 min per participant

**Evaluation Dimensions:**

| Dimension | Question | Weight |
|---|---|---|
| **Visual Quality** | The images are clear, detailed, and well-composed | 20% |
| **Narration Clarity** | The narration is clear and easy to understand | 15% |
| **Narrative Coherence** | The story is coherent and makes sense | 20% |
| **Music-Mood Fit** | The music matches the emotional tone of the story | 20% |
| **Audio-Visual Sync** | The narration and images are well-synchronized | 15% |
| **Overall Satisfaction** | I would use this tool to generate concept videos | 10% |

**Baseline Comparison:**
- Generate 5 identical prompts using manual pipeline: separate tools, combined by hand in MoviePy
- No emotion propagation, no intelligent timing (just 12s per scene)
- Compare DreamScapeAI scores vs. manual baseline scores (paired t-test)

**Output:**
- Mean Opinion Score (MOS) per dimension: `MOS = mean(all ratings) ± SD`
- Hypothesis test: MOS(DreamScapeAI) – MOS(Manual) > 0.5 on Coherence
- Fleiss' kappa for inter-rater agreement (target ≥0.60)

---

## 12. Project Timeline (20 Weeks)

Revised from 16 to 20 weeks to accommodate realistic human evaluation and thesis writing.

| Phase | Weeks | Deliverable | Acceptance Criteria |
|---|---|---|---|
| **1. Lit Review & Gap Analysis** | 1–2 | Research report: multimodal generation + synchronization methods | 15–20 citations; identified gap (no open coordination layer) |
| **2. Environment Setup** | 3 | Models downloaded; GPU access verified; FastAPI + Gradio skeleton | SDXL, XTTS, MusicGen load without OOM; API responds to test request |
| **3. Stage Implementation (Serial)** | 4–9 | Stages 1–6 working independently | Each stage accepts valid input, produces valid output; timing logs captured |
| **3a. Prompt Parsing & Scene Expansion** | 4–5 | LLM prompt engineering for scene structure | Generates valid JSON; 4–8 scenes per prompt |
| **3b. Visual Generation** | 5–6 | SDXL integration + negative prompt logic | 1024×576 images, <40s per image on T4 |
| **3c. Narration + Subtitles** | 6–7 | XTTS-v2 + Whisper integration | Per-scene WAV files + SRT with timestamps <200ms error |
| **3d. Music Generation** | 7–8 | MusicGen setup + conditioning | 90s+ audio track matching video duration |
| **3e. Video Assembly** | 8–9 | MoviePy + FFmpeg pipeline | Valid MP4, H.264, 1080p, 30 fps; audible narration + music |
| **4. Orchestration Layer** | 9–11 | TTS-first timing, emotion propagation, caching, error handling | Integrated end-to-end pipeline; Stage 1 → MP4 in <10 min |
| **5. UI (Gradio MVP)** | 11–12 | Gradio demo: prompt input, scene preview, video output | Deployed on HF Spaces; <10 min generation time |
| **6. Evaluation Setup** | 12 | Evaluation rubric, rater instructions, baseline videos | 5 manual videos prepared; rubric reviewed by advisor |
| **7. Human Study Execution** | 13–15 | Recruitment, data collection, analysis | 20+ participants; 200+ ratings; kappa calculated |
| **8. Benchmark & Refinement** | 15–16 | Automated metrics on 100-prompt test set; bugfixes | Success rate ≥85%; MOS per dimension reported |
| **9. Thesis Writing** | 17–19 | Draft thesis: intro, methods, results, discussion | 50–70 pages; figures for architecture, example outputs, results |
| **10. Defense & Polish** | 20 | Final thesis, slides, live demo | Passed defense; code + models published |

**Risk Buffers:** Weeks 16–20 contain 4-week buffer for thesis writing overruns, late human study recruitment, or model retraining.

---

## 13. Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation Strategy |
|---|---|---|---|
| **GPU quota exhausted on HF Spaces (T4 timeout)** | High | High | Use Colab Pro ($12/month) or RunPod ($15/hr for large jobs) as backup; cache aggressively; log quota warnings. Fallback: reduce image resolution to 512×384 (faster). |
| **SDXL outputs inconsistent across scenes (no character continuity)** | High | Medium | v1 explicitly non-goal; document in user stories. v2 will use IP-Adapter + seed control. Add UI note: "Each scene is independently generated." |
| **Generation time exceeds 10 minutes** | Medium | Medium | Profile each stage. SDXL is bottleneck (~3–4 min for 6 scenes). Options: (a) accept 12 min as realistic, (b) use SDXL-Turbo (4 steps, faster, lower quality), (c) reduce image count to 4 scenes. Choose (a) for v1.1. |
| **TTS sounds robotic** | Medium | Medium | XTTS-v2 acceptable for research; document in thesis ("not production-grade"). Offer ElevenLabs as optional paid tier in v2. |
| **Music doesn't match emotional tone** | Medium | Low | Test MusicGen conditioning on 10 prompts; iterate prompt templates. Fallback: generic ambient music if conditioning fails. |
| **NSFW/copyright outputs slip through** | Low | High | Strict keyword blocklist (40+ terms) + Stability AI safety checker. Manual review of all 10 study videos. Log all flagged outputs. |
| **Human study under-recruited** | Medium | High | Start recruitment in week 12 (not week 13); offer $10 gift card incentive; recruit from Discord/Reddit ML communities, university Slack. Target 20 (pilot), stretch to 30. |
| **Thesis advisor wants v2 features in v1** | Medium | Medium | Have written decision doc ready: "Non-Goals for v1" section. Propose v2 roadmap (character consistency, per-scene music, lip-sync). |
| **Whisper timestamps drift across scenes** | Low | Medium | Validate SRT timing in 5 videos manually; if drift >200ms, use Whisper's word-level alignment + force-align against narration text. |
| **Models don't fit in HF Spaces GPU memory simultaneously** | Low | High | Architecture is sequential by design (not parallel). If OOM during assembly, offload image display to CPU. Tested in week 3 (environment setup). |

---

## 14. Open Questions (Resolved in v1.1)

| Question | Decision for v1.1 | Rationale |
|---|---|---|
| Per-scene music or one track? | **One track for whole video** | Performance constraint (MusicGen ~60–90s per track). Per-scene deferred to v2. |
| Character consistency in v1? | **No; explicitly non-goal** | Requires IP-Adapter + seed control (~30s overhead per scene). Documented as v2 feature. |
| API models (GPT-4o) or local (Llama)? | **Local Llama-3.1-8B** | Free, reproducible, no rate limits. API swap available in v2 for users who want higher coherence. |
| Lip-sync animation? | **No; deferred to v2** | Requires separate image-to-video stage (SVD, AnimateDiff). Too much scope for v1. |
| Speed (Turbo) or quality (full SDXL)? | **Full SDXL with 10-min realistic timeline** | Turbo sacrifices visual quality noticeably. Accept 10 min vs. 5 min for better research quality. |
| How to measure "narrative coherence" objectively? | **5-point Likert + Fleiss' kappa for inter-rater reliability** | Standard in HCI/NLP evaluation. Documented in Section 11.2. |
| Storage for models on GitHub? | **No; use HF Hub** | SDXL + MusicGen = 12–15 GB; too large for GitHub. Publish code on GitHub; models on HF Hub; datasets on HF Datasets. |

---

## 15. Success Criteria (Definition of Done)

The project is **complete** when **all** of the following are satisfied:

- [ ] **Functional**: A user can enter a prompt (≤500 chars) on the deployed Gradio app and receive a valid MP4 within 10 minutes
- [ ] **Reliability**: At least 85 of 100 test prompts produce valid output (success rate ≥85%)
- [ ] **Quality**: Mean Opinion Score (MOS) ≥3.5/5 on narrative coherence in human evaluation study (n≥20)
- [ ] **Sync**: Subtitle-to-audio sync error <200 ms (automated measurement + spot-checks)
- [ ] **Safety**: Zero high-severity NSFW/violence outputs in final evaluation dataset (manual audit)
- [ ] **Research**: All four research contributions documented in thesis with quantitative + qualitative evidence
- [ ] **Reproducibility**: Same prompt + seed → identical output (tested on 10 prompts)
- [ ] **Code**: All code published on GitHub (GPL v3 or MIT); documented setup instructions
- [ ] **Models**: Model weights published on HF Hub with license; evaluation rubric on HF Hub as PDF
- [ ] **Data**: Evaluation dataset (prompts + ratings) published on HF Datasets (anonymized)
- [ ] **Demo**: Live demo on HF Spaces, responsive UI, keyboard-accessible
- [ ] **Thesis**: 50–70 page document with figures, results tables, inter-rater kappa, paired t-test results; defended successfully

---

## 16. Future Work (v2 Roadmap)

### 16.1 Character Consistency
- Integrate **IP-Adapter** or **AnimateDiff** for seed-controlled, consistent character across scenes
- Cost: ~30s overhead per scene; acceptable for v2

### 16.2 Per-Scene Music
- Generate independent music track per scene, then crossfade
- Better emotional reactivity; higher compute cost (~2x slower)

### 16.3 Lip-Sync & Video Synthesis
- Use **Stable Video Diffusion (SVD)** or **AnimateDiff** to animate still images with realistic mouth movement
- Aligns narration timing to lip motion automatically

### 16.4 Multi-Speaker & Dialogue
- Detect dialogue in prompt; assign different voices to characters
- Use voice cloning (speaker embedding) for character consistency

### 16.5 API Tier with ElevenLabs & GPT-4o
- Premium tier: GPT-4o for scene expansion (higher coherence), ElevenLabs for TTS (production-grade voice)
- Cost: ~$0.30–0.50 per video
- Free tier remains local

### 16.6 Advanced UI (React)
- Real-time progress tracking per stage
- Advanced regeneration workflow (edit scenes, regenerate independently)
- Video player with in-frame annotations (show scene boundaries, sync error diagnostics)

### 16.7 Longer Videos (Up to 5 Minutes)
- Current pipeline caches outputs; extend to multi-part videos
- May require batching architecture (queue system) for HF Spaces

---

## 17. References & Related Work

### Foundational Papers
- **Make-A-Video** (Singer et al., Meta 2022) — Text-to-video via temporal latent diffusion; baseline for video generation quality
- **Stable Video Diffusion** (Blattmann et al., Stability AI 2023) — Image-to-video; animation baseline
- **AudioLDM2** (Liu et al., Meta 2023) — Text-to-audio generation; music conditioning methods
- **MusicGen** (Copet et al., Meta 2023) — State-of-the-art open music generation; core model for v1
- **XTTS-v2** (Casanova et al., Coqui 2023) — Multilingual TTS with speaker control; used in narration stage
- **Whisper** (Radford et al., OpenAI 2022) — Robust ASR; used for subtitle generation

### Multimodal Synchronization & Coherence
- **VideoPoet** (Kondraciuk et al., Google 2024) — Unified multimodal LLM for video generation; closed-source reference
- **Sora** (OpenAI 2024) — Text-to-video with superior visual quality; closed-source, not accessible; reference only
- **Synchronizing Multimodal Semantic Graphs** (Chen et al., 2022) — Graph-based alignment approach; alternative to our TTS-first method

### Emotion & Sentiment in Multimedia
- **Affective Computing in Multimedia** (Hanjalic & Xu, IEEE 2005) — Foundational work on emotion modeling
- **EmoMusicGen** (Vázquez-Rodríguez et al., 2023) — Emotion conditioning for music generation; directly applicable

### Evaluation Methodologies
- **Defining Narrative Coherence** (Graesser et al., 1997) — Cognitive science approach to measuring coherence
- **Mean Opinion Score (MOS) in Multimedia** (ITU-R BS.1534) — Standard evaluation protocol for audio/video quality

---

## 18. Appendix A: Glossary

| Term | Definition |
|---|---|
| **Orchestrator** | Central Python class coordinating all pipeline stages; manages caching, error recovery, temporal alignment |
| **TTS-First Timing** | Architecture decision: narration audio duration drives image display duration (not the reverse) |
| **Emotion Propagation** | Process of extracting sentiment from prompt and applying it to SDXL negative prompt, XTTS speaker params, and MusicGen conditioning |
| **CLIPScore** | Metric measuring image-text alignment via CLIP embeddings; ranges 0–1 (higher = better alignment) |
| **Whisper WER** | Word Error Rate; measures ASR accuracy; calculated as (substitutions + deletions + insertions) / total words |
| **Sync Error** | Time offset between subtitle and audio, measured via cross-correlation; target <200 ms |
| **MOS** | Mean Opinion Score; average Likert rating across all raters; 1–5 scale |
| **Fleiss' Kappa** | Inter-rater agreement metric for categorical data; 0–1 scale (≥0.60 acceptable) |
| **VRAM** | Video Random Access Memory; GPU memory for model weights + intermediate activations |
| **HF Spaces** | Hugging Face Spaces; free cloud GPU tier (T4) for hosting interactive demos |
| **ZeroGPU** | HF Spaces zero-cost GPU tier; limited throughput, 5-minute per execution timeout |
| **LoRA** | Low-Rank Adaptation; efficient model fine-tuning technique; not used in v1 |
| **IP-Adapter** | Image Prompt Adapter; enables character consistency across images; v2 feature |
| **SDXL Turbo** | Faster Stable Diffusion variant (4 steps vs. 50 steps); lower quality; v2 option |

---

## 19. Appendix B: Example Evaluation Rubric (Rater Instructions)

**Video Evaluation Form**

*You will watch a 60-second AI-generated video. Rate it on each dimension using the scale below.*

**Scale:** 1 = Strongly Disagree | 2 = Disagree | 3 = Neutral | 4 = Agree | 5 = Strongly Agree

**1. Visual Quality** (Clarity, composition, absence of artifacts)
- "The images in this video are clear, well-composed, and visually appealing."
  - [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**2. Narration Clarity** (Audio quality, intelligibility)
- "The narration is clear and easy to understand."
  - [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**3. Narrative Coherence** (Story makes sense, logical progression)
- "The story in this video is coherent and makes logical sense."
  - [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**4. Music-Mood Fit** (Music matches emotional tone)
- "The background music matches the emotional tone of the story."
  - [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**5. Audio-Visual Synchronization** (Images match narration timing)
- "The images are well-synchronized with the narration. There are no awkward delays or mismatches."
  - [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**6. Overall Satisfaction**
- "I would find this tool useful for generating concept videos."
  - [ ] 1 [ ] 2 [ ] 3 [ ] 4 [ ] 5

**Optional Comments:**
```
[Text field for qualitative feedback]
```

---

## 20. Appendix C: Hardware & GPU Requirements

### Development Environment (Week 3 Setup Checklist)
- [ ] **CPU:** 8+ cores, 16GB+ RAM (for running Ollama + one model at a time)
- [ ] **GPU:** NVIDIA T4 or better, ≥12GB VRAM (tested stack below)
- [ ] **Storage:** 50 GB+ free disk (for model weights + cache)
- [ ] **CUDA:** 12.x compatible
- [ ] **Python:** 3.10+

### Model VRAM Footprint (Worst Case)
| Model | VRAM | Notes |
|---|---|---|
| Llama-3.1-8B (Ollama) | 2–4 GB | Quantized; runs on CPU OK if needed |
| SDXL (diffusers, fp32) | 10–12 GB | Use fp16 / int8 quantization if OOM |
| XTTS-v2 | 4–6 GB | GPU inference + speaker embeddings |
| MusicGen-medium | 6–8 GB | GPU inference |
| Whisper (base) | 2–3 GB | Smaller than large, sufficient accuracy |
| **Total (sequential)** | 12–15 GB peak | All fit on single T4 (16GB) if sequential; parallel = OOM |

### Benchmarks on T4 GPU (HF Spaces)

| Stage | Avg Time (T4) | Range | Bottleneck |
|---|---|---|---|
| Scene Expansion (Stage 2) | 15s | 10–20s | LLM token generation |
| Image Generation (Stage 3) | 30–40s per image | 25–60s | SDXL denoising steps (50 steps) |
| Narration (Stage 4) | 20s per 30s narration | 15–30s | XTTS inference |
| Subtitles (Stage 5) | 8s per 30s audio | 5–15s | Whisper inference |
| Music (Stage 6) | 70s per 90s track | 60–90s | MusicGen generation |
| Video Assembly (Stage 7) | 20s | 15–30s | FFmpeg encoding (H.264) |
| **Total (6 scenes, 60s video)** | ~5–7 min | 4–10 min | SDXL bottleneck (3–4 min) |

**Note:** HF Spaces T4 is shared and throttled; expect +20% variance. Use caching to offset repeated stages.

---

*End of PRD v1.1 — Ready for Implementation*
