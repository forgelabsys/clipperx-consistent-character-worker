# ClipperX Consistent Character Worker

RunPod Serverless worker: SDXL + IP-Adapter, for generating stickman-style
scene images that keep a reference character's face/hair/clothing consistent
across many generations — the automation approach discussed for scaling the
Galloping Gertie-style pipeline without the ~4min/image cost of the Flux2
edit-mode worker.

## Input

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Scene description (style anchor + action) |
| `reference_image_base64` | str | required | Character reference image, base64 (no `data:` prefix) |
| `negative_prompt` | str | generic quality negatives | What to avoid |
| `ip_adapter_scale` | float | `0.6` | How strongly the reference identity is enforced (0-1) |
| `num_inference_steps` | int | `28` | Denoising steps |
| `guidance_scale` | float | `6.0` | Prompt adherence |
| `width` / `height` | int | `1024` / `1024` | Output resolution |
| `seed` | int | random | For reproducibility |

## Output

```json
{ "image": "<base64 PNG>" }
```

## Notes

- Model weights are baked into the Docker image at build time (no runtime
  download), so cold starts only pay for container boot + model load into
  VRAM, not a multi-GB HF Hub download.
- Needs a real GPU with enough VRAM for SDXL fp16 (16GB+ recommended).
