import torch
from torch import nn

# Stable Diffusion NSFW filter config (see LAION-AI/CLIP-based-NSFW-Detector repo).
concepts = ['sexual', 'nude', 'sex', '18+', 'naked', 'nsfw', 'porn', 'dick', 'vagina',
            'naked child', 'explicit content', 'uncensored', 'fuck', 'nipples', 'visible nipples', 'naked breasts', 'areola']
special_concepts = ["little girl", "young child", "young girl"]


def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())


@torch.no_grad()
def forward_inspect(self, clip_input, images):
    pooled_output = self.vision_model(clip_input)[1]
    image_embeds = self.visual_projection(pooled_output)

    special_cos_dist = cosine_distance(
        image_embeds, self.special_care_embeds
    ).cpu().numpy()
    cos_dist = cosine_distance(image_embeds, self.concept_embeds).cpu().numpy()

    matches = {"nsfw": [], "special": []}
    batch_size = image_embeds.shape[0]
    for i in range(batch_size):
        result_img = {
            "special_scores": {}, "special_care": [], "concept_scores": {}, "bad_concepts": []
        }

        adjustment = 0.0

        for concet_idx in range(len(special_cos_dist[0])):
            concept_cos = special_cos_dist[i][concet_idx]
            concept_threshold = self.special_care_embeds_weights[concet_idx].item(
            )
            result_img["special_scores"][concet_idx] = round(
                concept_cos - concept_threshold + adjustment, 3
            )
            if result_img["special_scores"][concet_idx] > 0:
                result_img["special_care"].append(
                    {concet_idx, result_img["special_scores"][concet_idx]}
                )
                adjustment = 0.01
                matches["special"].append(special_concepts[concet_idx])

        for concet_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[i][concet_idx]
            concept_threshold = self.concept_embeds_weights[concet_idx].item()
            result_img["concept_scores"][concet_idx] = round(
                concept_cos - concept_threshold + adjustment, 3
            )

            if result_img["concept_scores"][concet_idx] > 0:
                result_img["bad_concepts"].append(concet_idx)
                matches["nsfw"].append(concepts[concet_idx])

    has_nsfw_concepts = len(matches["nsfw"]) > 0

    return matches, has_nsfw_concepts
