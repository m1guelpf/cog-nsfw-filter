<h1 align="center">Stable Diffusion NSFW Filter</h1>
<p align="center"><b><a href="https://nsfw.m1guelpf.me" target="_blank">Try it out</a> | <a href="https://replicate.com/m1guelpf/nsfw-filter" target="_blank">View on Replicate</a></b></p>


This repo contains a modified implementation of the example code provided in the [Red-Teaming the Stable Diffusion Safety Filter](https://arxiv.org/abs/2210.04610v5) paper.

## Development

> **Note** If you just wanna try the model out or run it in production, see the link above.

This model is packaged as a Cog model, a tool to packages machine learning models as standard containers.

First, [download Cog](https://github.com/replicate/cog#install) on your system. Then, download the pre-trained weights [with your Hugging Face auth token](https://huggingface.co/settings/tokens):

    cog run script/download-weights <your-hugging-face-auth-token>

Once setup, you can run predictions:

    cog predict -i image=@/path/to/image.jpg
