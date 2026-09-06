# ClipperX Consistent Character Worker

RunPod Serverless worker: **FLUX.2 [klein] 4B** (Apache 2.0, distilled for
speed — as low as 4 inference steps), which has *native* multi-reference
image editing support. This replaced an earlier SDXL + IP-Adapter attempt
that proved unstable for our flat-vector stickman art style (IP-Adapter is a
bolt-on adapter trained mostly on photos; this model was trained end-to-end
to do reference-conditioned editing, so it should hold identity far more
reliably).

## Input

| Field | Type | Default | Description |
|---|---|---|---|
| `prompt` | str | required | Scene description / edit instruction |
| `reference_image_base64` | str | optional | Single reference image, base64 (no `data:` prefix) |
| `reference_images_base64` | str[] | optional | Multiple reference images (multi-reference editing) |
| `num_inference_steps` | int | `4` | Denoising steps (the model is distilled for very few) |
| `guidance_scale` | float | `1.0` | Prompt adherence |
| `width` / `height` | int | `1024` / `1024` | Output resolution (ignored in editing mode unless both are set) |
| `seed` | int | random | For reproducibility |

If no reference image is given, it runs as plain text-to-image. If one or
more are given, it runs as image editing (the reference character/scene is
redrawn per the prompt).

## Output

```json
{ "image": "<base64 PNG>" }
```

## Notes

- Model weights are baked into the Docker image at build time.
- Needs a real GPU (13GB+ VRAM recommended per the model card).
