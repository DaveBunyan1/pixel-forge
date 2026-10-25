from collections.abc import Callable
from typing import Any

import torch
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import (
    StableDiffusionXLPipeline,
)
from PIL import Image

StepCallback = Callable[
    [
        StableDiffusionXLPipeline,
        int,
        torch.Tensor,
        dict[str, Any],
    ],
    dict[str, Any],
]

model = "SG161222/RealVisXL_V5.0"


class TextToImage:
    pipe: StableDiffusionXLPipeline | None = None

    def load_model(self) -> None:
        if torch.cuda.is_available():
            device = "cuda"
            dtype = torch.float16
        elif torch.backends.mps.is_available():
            device = "mps"
            dtype = torch.float16
        else:
            device = "cpu"
            dtype = torch.float32

        print(f"Using {device.upper()} with {dtype}")

        pipe = StableDiffusionXLPipeline.from_pretrained(
            model,
            dtype=dtype,
            use_safetensors=True,
        )
        pipe.enable_model_cpu_offload()
        self.pipe = pipe

    def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str | None = None,
        num_steps: int = 30,
        callback: StepCallback | None = None,
    ) -> Image.Image:
        if self.pipe is None:
            raise RuntimeError("Pipeline is not loaded")

        result = self.pipe(
            prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_steps,
            guidance_scale=9.0,
            callback_on_step_end=callback,
        )
        return result.images[0]


if __name__ == "__main__":
    text_to_image = TextToImage()
    text_to_image.load_model()

    def callback(
        pipe: StableDiffusionXLPipeline,
        step: int,
        timestep: torch.Tensor,
        callback_kwargs: dict[str, Any],
    ) -> dict[str, Any]:
        print(f"🚀 Step {step + 1}/30")
        return callback_kwargs

    image = text_to_image.generate(
        "A Renaissance castle in the Loire Valley",
        negative_prompt="low quality, ugly",
        callback=callback,
    )

    image.save("output.png")
