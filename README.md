# Stable Diffusion NSFW Filter

[![Replicate](https://replicate.com/m1guelpf/nsfw-filter/badge)](https://replicate.com/m1guelpf/nsfw-filter)

A modified version of [Stable Diffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5)'s NSFW as a Cog model. [Cog packages machine learning models as standard containers.](https://github.com/replicate/cog)

First, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run script/download-weights <your-hugging-face-auth-token>

Then, you can run predictions:

    cog predict -i image=@/path/to/image.jpg
