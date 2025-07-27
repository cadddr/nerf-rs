use rand::prelude::StdRng;
use rand::SeedableRng;

pub use dfdx::arrays::HasArrayData;
use dfdx::devices::{Cpu, HasDevice};
use dfdx::gradients::{
    CanUpdateWithGradients, GradientProvider, Gradients, OwnedTape, Tape, UnusedTensors,
};
use dfdx::losses::{
    binary_cross_entropy_with_logits_loss, cross_entropy_with_logits_loss, mse_loss,
};
use dfdx::nn::{Linear, Module, ReLU, ResetParams, Sigmoid, Tanh};
pub use dfdx::optim::{Adam, AdamConfig, Momentum, Optimizer, Sgd, SgdConfig, WeightDecay};
pub use dfdx::tensor::{tensor, Tensor, Tensor0D, Tensor1D, Tensor2D, TensorCreator};
use dfdx::tensor_ops::{add, backward, sub};

pub const BATCH_SIZE: usize = 16384;

pub type MLP = (
    //TODO: 8 layers 256 each
    //    the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer)
    //    and outputs σ and a 256-dimensional feature vector.
    //    This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels)
    //    that output the view-dependent RGB color.
    (Linear<2, 100>, Tanh),
    (Linear<100, 100>, Tanh),
    (Linear<100, 100>, Tanh),
    (Linear<100, 100>, Tanh),
    (Linear<100, 100>, Tanh),
    (Linear<100, 4>),
);

pub struct DfdxMlp {
    mlp: MLP,
    opt: Adam<MLP>,
}

impl DfdxMlp {
    pub fn new() -> DfdxMlp {
        let (mut mlp, mut opt): (MLP, Adam<MLP>) = init_mlp();
        DfdxMlp { mlp: mlp, opt: opt }
    }

    pub fn step(&mut self, y: Tensor2D<BATCH_SIZE, 4, OwnedTape>, gold: Vec<[f32; 4]>) -> f32 {
        step(&mut self.mlp, &mut self.opt, y, gold)
    }

    pub fn predict(
        &self,
        coords: Vec<[f32; 2]>,
        views: Vec<[f32; 3]>,
        points: Vec<[f32; 3]>,
    ) -> Tensor2D<BATCH_SIZE, 4, OwnedTape> {
        predict_emittance_and_density(&self.mlp, coords, views, points)
    }

    pub fn BATCH_SIZE(&self) -> usize {
        BATCH_SIZE
    }

    pub fn get_predictions_as_array_vec(
        &self,
        predictions: &Tensor2D<BATCH_SIZE, 4, OwnedTape>,
    ) -> [[f32; 4]; BATCH_SIZE] {
        *predictions.data()
    }
}

pub fn init_mlp() -> (MLP, Adam<MLP>) {
    // Rng for generating model's params
    let mut rng = StdRng::seed_from_u64(0);

    let mut mlp: MLP = Default::default();
    mlp.reset_params(&mut rng);

    let mut opt: Adam<MLP> = Adam::new(AdamConfig {
        lr: 5e-5,
        betas: [0.5, 0.25],
        eps: 1e-6,
        weight_decay: Some(WeightDecay::Decoupled(1e-2)),
    });

    return (mlp, opt);
}

pub fn step(
    model: &mut MLP,
    opt: &mut Adam<MLP>,
    y: Tensor2D<BATCH_SIZE, 4, OwnedTape>,
    gold: Vec<[f32; 4]>,
) -> f32 {
    // compute cross entropy loss
    let y_true: Tensor2D<BATCH_SIZE, 4> =
        tensor(array_vec_to_2d_array::<[f32; 4], BATCH_SIZE>(gold));
    let loss: Tensor0D<OwnedTape> = mse_loss(y, y_true);
    let loss_data: f32 = loss.data().clone();
    // call `backward()` to compute gradients. The tensor *must* have `OwnedTape`!
    let gradients: Gradients = loss.backward();
    // pass the gradients & the model into the optimizer's update method
    opt.update(model, gradients);

    return loss_data;
}

use std::convert::TryInto;

fn array_vec_to_2d_array<T, const N: usize>(v: Vec<T>) -> [T; N] {
    v.try_into()
        .unwrap_or_else(|v: Vec<T>| panic!("Expected a Vec of length {} but it was {}", N, v.len()))
}

#[test]
fn test_array_vec_to_2d_array() {
    let mut v: Vec<[f32; 3]> = Vec::new();
    for i in 0..3 {
        v.push([1., 2., 3.]);
    }
    println!("{:?}", array_vec_to_2d_array::<[f32; 3], 3>(v));
}

pub fn predict_emittance_and_density(
    mlp: &MLP,
    coords: Vec<[f32; 2]>,
    views: Vec<[f32; 3]>,
    points: Vec<[f32; 3]>,
) -> Tensor2D<BATCH_SIZE, 4, OwnedTape> {
    //    let mut predictions: Vec<Tensor1D<4, OwnedTape>> = Vec::new();
    //TODO: also use view directions
    //    for point in views {
    //        let x: Tensor1D<3> = tensor(*point);
    //        let y = mlp.forward(x.trace());
    //        predictions.push(y);
    //    }
    //    let x: Tensor2D<BATCH_SIZE, 3> = tensor(array_vec_to_2d_array::<[f32; 3], BATCH_SIZE>(views));
    let x: Tensor2D<BATCH_SIZE, 2> = tensor(array_vec_to_2d_array::<[f32; 2], BATCH_SIZE>(coords));
    //    let x: Tensor2D<BATCH_SIZE, 3> = dev.stack(views.iter().map(|x| tensor(*x)));

    let mut predictions: Tensor2D<BATCH_SIZE, 4, OwnedTape> = mlp.forward(x.trace());

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

pub fn accumulate_radiance(
    predictions: Vec<Tensor1D<4, OwnedTape>>,
) -> Vec<Tensor1D<3, OwnedTape>> {
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
        sub(
            predictions[r].with_empty_tape(),
            predictions[r + 1].with_empty_tape(),
        );
    }

    return radiances;
}

pub fn from_u8_rgb(r: u8, g: u8, b: u8) -> u32 {
    let (r, g, b) = (r as u32, g as u32, b as u32);
    (r << 16) | (g << 8) | b
}

pub fn rgba_to_u8_array(rgba: &u32) -> [u8; 3] {
    [(*rgba >> 16) as u8, (*rgba >> 8) as u8, *rgba as u8]
}

pub fn prediction_as_u32(prediction: &Tensor1D<4, OwnedTape>) -> u32 {
    let rgba: &[f32; 4] = prediction.data();
    return from_u8_rgb(
        (rgba[0] * 255.) as u8,
        (rgba[1] * 255.) as u8,
        (rgba[2] * 255.) as u8,
    );
}

pub fn prediction_array_as_u32(rgba: &[f32; 4]) -> u32 {
    return from_u8_rgb(
        (rgba[0] * 255.) as u8,
        (rgba[1] * 255.) as u8,
        (rgba[2] * 255.) as u8,
    );
}
