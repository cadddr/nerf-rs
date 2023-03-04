use rand::prelude::StdRng;
use rand::{SeedableRng};

use dfdx::nn::{Linear, ReLU, Tanh, Sigmoid, ResetParams, Module};
pub use dfdx::tensor::{tensor, Tensor, Tensor0D, Tensor1D, Tensor2D, TensorCreator};
pub use dfdx::arrays::HasArrayData;
use dfdx::gradients::{Gradients, CanUpdateWithGradients, GradientProvider, OwnedTape, Tape, UnusedTensors};
use dfdx::tensor_ops::{add, sub, backward};
use dfdx::losses::{cross_entropy_with_logits_loss, mse_loss, binary_cross_entropy_with_logits_loss};
pub use dfdx::optim::{Sgd, SgdConfig, Optimizer, Momentum};

const BATCH_SIZE: usize = 32;

pub type MLP = (
        //TODO: 8 layers 256 each
//    the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer)
//    and outputs σ and a 256-dimensional feature vector.
//    This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels)
//    that output the view-dependent RGB color.
    (Linear<3, 32>, ReLU),
    (Linear<32, 32>, ReLU),
(Linear<32, 32>, ReLU),
(Linear<32, 32>, ReLU),
(Linear<32, 32>, ReLU),
    (Linear<32, 4>, Sigmoid),
);

pub fn init_mlp() -> (MLP, Sgd<MLP>) {
    // Rng for generating model's params
    let mut rng = StdRng::seed_from_u64(0);

    let mut mlp: MLP = Default::default();
    mlp.reset_params(&mut rng);

    // Use stochastic gradient descent (Sgd), with a learning rate of 1e-2, and 0.9 momentum.
    let mut opt: Sgd<MLP> = Sgd::new(SgdConfig {
        lr: 2e-2,
        momentum: Some(Momentum::Classic(0.9)),
        weight_decay: None,
    });

    return (mlp, opt)
}

pub fn step(model: &mut MLP, opt: &mut Sgd<MLP>, y: Tensor1D<4, OwnedTape>, y_true: Tensor1D<4>) -> f32 {
    // compute cross entropy loss
    let loss: Tensor0D<OwnedTape> = mse_loss (y, y_true);
    let loss_data: f32 = loss.data().clone();
    // call `backward()` to compute gradients. The tensor *must* have `OwnedTape`!
    let gradients: Gradients = loss.backward();
    // pass the gradients & the model into the optimizer's update method
    opt.update(model, gradients);

    return loss_data;
}

pub fn predict_emittance_and_density(mlp: &MLP, views: &Vec<[f32; 3]>, points: &Vec<[f32; 3]>) -> Vec<Tensor1D<4, OwnedTape>> {
    let mut predictions: Vec<Tensor1D<4, OwnedTape>> = Vec::new();
    //TODO: also use view directions
    for point in points {
        let x: Tensor1D<3> = tensor(*point);
        let y = mlp.forward(x.trace());
        predictions.push(y);
    }

    return predictions;

    //
    //    // Construct model
    //    let mut model: Mlp<10, 512, 10> = Mlp::default();
    //    model.reset_params(&mut rng);
    //
    //    // Forward pass with a single sample
    //    let sample: Tensor1D<10> = Tensor1D::randn(&mut rng);
    //    let result: Tensor1D<10> = model.forward(sample);
    //    println!("{:?}", result);
    //
    //    // Forward pass with a batch of samples
    //    let batch: Tensor2D<BATCH_SIZE, 10> = Tensor2D::randn(&mut rng);
    //    let _: Tensor2D<BATCH_SIZE, 10, OwnedTape> = model.forward(batch.trace());
}

pub fn accumulate_radiance(predictions: Vec<Tensor1D<4, OwnedTape>>) ->  Vec<Tensor1D<3, OwnedTape>> {
    // Cˆ(r)=\sum{i=1}{N} Ti(1−exp(−σiδi))ci,
    //    where Ti =exp(−\sum{j=1}{i-1}σjδj),
    //where δi = ti+1 − ti is the distance between adjacent samples.
    let radiances: Vec<Tensor1D<3, OwnedTape>> = Vec::new(); //one per each ray
    //NUM_SAMPLES points for each ray are concatenated together
    //TODO: should be sample distances not predictions (rgba)
    for r in (0..predictions.len()).step_by(1) {
        //    for prediction in predictions {
        //        let mut samples: Vec<Tensor1D<4, OwnedTape>> = Vec::new();  //predictions[r..r + NUM_SAMPLES].try_into().unwrap();
        //        for i in (r..r + NUM_SAMPLES) {
        //            samples.push(predictions[i]);
        //        }
        //        let sample_distances: Vec<Tensor0D<OwnedTape>> = Vec::new();//samples.iter().map(|&it| vec3_len(it)).collect::<Vec<f32>>();;
        //        for i in (0..samples.len() - 1) {
        //            let dist = sub(samples[i + 1], samples[i]);
        //        }
        sub(predictions[r].with_empty_tape(), predictions[r+1].with_empty_tape());
    }



    return radiances;
}

pub fn from_u8_rgb(r: u8, g: u8, b: u8) -> u32 {
    let (r, g, b) = (r as u32, g as u32, b as u32);
    (r << 16) | (g << 8) | b
}

pub fn prediction_as_u32(prediction: &Tensor1D<4, OwnedTape>) -> u32 {
    let rgba: &[f32; 4] = prediction.data();
    return from_u8_rgb((rgba[0] * 255.) as u8, (rgba[1] * 255.) as u8, (rgba[2] * 255.) as u8)
}