### Paper Implementation
# NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
[https://arxiv.org/pdf/2003.08934]

#### Basic approach:
For a given scene, an MLP predicts the radiance density distribution at random points along rays traced from a particular view point into the scene.
When rendering a novel view, density estimates at those query points are composited to produce the final pixel color.

#### Implementation details:
- `main::get_batch` We iterate over batches of pixels from renderings of different viewing angles.
- `ray_sampling::sample_points_tensor_along_view_directions` For a random batch of pixel coordinates and a viewing angle, sample points along rays through those pixels.
- `model_tch::predict` Run those query points through an MLP to estimate 4D radiance values at each point.
- `model_ch::compositing` Integrates radiance estimates at query points into the final pixel color resulting from that ray.
- `model_tch::step` Backprop MSE loss vs. gold pixels.


#### Dependencies:
- `torch` for deep learning.
- `tensorboard` for logging.

#### Demo:


// TODO:
/*
- sample highest error samples
- predict shading as different channels
*/
