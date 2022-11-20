from typing import List

import torch
from PIL import Image
from functools import partial
from filter import forward_inspect
from cog import BasePredictor, BaseModel, Input, Path
from diffusers import StableDiffusionPipeline


MODEL_CACHE = "diffusers-cache"


class FilterOutput(BaseModel):
    nsfw_detected: bool
    nsfw: List[str]
    special: List[str]


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            cache_dir=MODEL_CACHE,
            local_files_only=True,
        ).to("cuda")

        self.pipe.safety_checker.forward = partial(
            forward_inspect, self=self.pipe.safety_checker
        )

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        image: Path = Input(
            description="Image to run through the NSFW filter",
        ),
    ) -> FilterOutput:
        """Run the provided image through the NSFW filter"""

        image = Image.open(image)
        safety_checker_input = self.pipe.feature_extractor(
            images=image, return_tensors="pt"
        ).to("cuda")

        result, has_nsfw_concepts = self.pipe.safety_checker.forward(
            clip_input=safety_checker_input.pixel_values, images=image
        )

        return FilterOutput(nsfw_detected=has_nsfw_concepts, nsfw=result["nsfw"], special=result["special"])
