# References

Papers this direction builds on. The PDFs are kept locally and are deliberately not
committed, since they are third party copyrighted work. Links point at the originals.

## Reconstruction based world models

The camp HERMES is testing against. These learn a latent by reconstructing observations,
on the theory that a model able to redraw a scene must understand it.

* **Mastering Diverse Domains through World Models** (DreamerV3). Hafner, Pasukonis, Ba,
  Lillicrap. https://arxiv.org/abs/2301.04104
* **Mastering Diverse Control Tasks through World Models**, and its ablation companion.
  The ablation paper is the more useful of the two here, since it isolates what the
  reconstruction term actually contributes.

## Decoder free and task centric world models

The camp HERMES is arguing for. These train the latent only on quantities that matter for
acting, with no pixel reconstruction anywhere.

* **Learning Massively Multitask World Models for Continuous Control** (TD-MPC2).
  Hansen, Su, Wang. https://arxiv.org/abs/2310.16828
* **Back to Parsimonious Latents: Learning Task Centric World Models from Visual
  Foundations.** The closest published statement of the HERMES thesis, and the most
  direct point of comparison for these experiments.

## Latent prediction and joint embedding

The self supervised middle ground, and the basis for arm D in Experiment 2.

* **Discrete JEPA.** Joint embedding predictive architecture with a discrete latent.
* **AquaJEPA: An Action Conditioned Multimodal JEPA Family for Underwater Robot
  Dynamics.** Useful mainly for its action conditioning treatment, which is the direction
  Experiment 3 is heading.

## Adjacent

* **Reflexion: Language Agents with Verbal Reinforcement Learning.** Not core to the
  world model question, kept for a separate line of reading.

## How these map onto the experiments

| paper family | arm in Experiment 2 |
| :--- | :--- |
| DreamerV3 and relatives | `A_recon`, task plus pixel reconstruction |
| TD-MPC2, parsimonious latents | `B_task`, task only with no decoder |
| JEPA family | `D_latent`, next latent prediction with no decoder |

Arm `C_scene`, which predicts background colour and distractor positions without ever
touching a pixel, has no direct analogue in the literature. It exists as a negative
control, to separate "reconstruction teaches the model about the scene" from the duller
"any second loss regularises the encoder".
