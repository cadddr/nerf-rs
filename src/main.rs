//We synthesize images by sampling 5D coordinates (location and viewing direction) along camera rays (a),
//feeding those locations into an MLP to produce a color and volume density (b),
//  We encourage the representation to be multiview consistent by restricting the network to predict the volume density σ as a function of only the location x, while allowing the RGB color c to be predicted as a function of both location and viewing direction. To accomplish this, the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer), and outputs σ and a 256-dimensional feature vector. This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.
//and using volume ren- dering techniques to composite these values into an image (c
//TODO: curious how it would do without relying on volume rendering approximation and density

use std;
use std::convert::TryInto;
use rand::prelude::StdRng;
use rand::{random, SeedableRng};
use vecmath::{vec3_add, vec3_sub, vec3_normalized, vec3_dot, vec3_cross, vec3_len, vec3_scale};
use dfdx::nn::{Linear, ReLU, Tanh, ResetParams, Module};
use dfdx::tensor::{tensor, Tensor, Tensor0D, Tensor1D, Tensor2D, TensorCreator};
use dfdx::gradients::{Gradients, CanUpdateWithGradients, GradientProvider, OwnedTape, Tape, UnusedTensors};
use dfdx::tensor_ops::{add, sub, backward};
use dfdx::losses::{cross_entropy_with_logits_loss, mse_loss};
use dfdx::optim::{Sgd, SgdConfig, Optimizer, Momentum};

mod mlp;
use mlp::Mlp;

const HITHER: f32 = 0.05;
const FOV: f32 =  std::f32::consts::PI / 4.;

const UP: [f32; 3] =   [0., 1.,  0.];
const FROM: [f32; 3] = [0., 0., -1.];
const AT: [f32; 3] =   [0., 0.,  1.];

fn screen_space_to_world_space(x: f32, y: f32, width: f32, height: f32) -> [f32; 3] {
    let off: f32 = f32::tan(FOV / 2.) * HITHER;
    let offset_left = off - 2. * off * x / width;
    let offset_up = off - 2. * off * y / height;

    let view = vec3_normalized(vec3_sub(AT, FROM));
    let left = vec3_normalized(vec3_cross(view, UP));

    let to = vec3_normalized(
            vec3_add(
                    vec3_add(
                            vec3_scale(view, HITHER),
                    vec3_scale(left, offset_left)
                    ),
            vec3_scale(UP, offset_up)
            )
    );

    return to;
}

const NUM_SAMPLES: usize = 1;
const RAY_PROB: f32 = 10./(512. * 512.);
const T_FAR: f32 = 10.;

fn sample_points_along_ray(from: [f32; 3], to: [f32; 3]) -> Vec<[f32; 3]> {
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut locations: Vec<f32> = Vec::new();
    for i in 0..NUM_SAMPLES {
        let t = random::<f32>() * T_FAR;
        let point = //vec3_add(from,
        vec3_scale(to, t);
        //);
        points.push(point);
        locations.push(t);
    }

    let mut both = points.into_iter().zip(locations.into_iter()).collect::<Vec<([f32; 3], f32)>>();
    both.iter().for_each(|it| {
        println!("unsorted {:?}", it);
    });

    both.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    both.iter().for_each(|it| {
        println!("sorted {:?}", it);
    });

    points = both.iter().map(|a| a.0).collect::<Vec<[f32; 3]>>();
    return points;
}

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn sample_points_along_view_directions() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    //returns xyz th, phi
    //TODO: returning view vectors rather than angles
    let mut views: Vec<[f32; 3]> = Vec::new();
    let mut points: Vec<[f32; 3]> = Vec::new();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if random::<f32>() <= RAY_PROB {
                //TODO: rewrite as vectorized
                let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
                views.push(to);
                points.append(&mut sample_points_along_ray(FROM, to));
            }
        }
    }
    return (views, points);
}

const BATCH_SIZE: usize = 32;

type MLP = (
        //TODO: 8 layers 256 each
//    the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer)
//    and outputs σ and a 256-dimensional feature vector.
//    This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional fully-connected layer (using a ReLU activation and 128 channels)
//    that output the view-dependent RGB color.
    (Linear<3, 32>, ReLU),
    (Linear<32, 32>, ReLU),
    (Linear<32, 4>, Tanh),
);

fn init_mlp() -> (MLP, Sgd<MLP>) {
    // Rng for generating model's params
    let mut rng = StdRng::seed_from_u64(0);

    let mut mlp: MLP = Default::default();
    mlp.reset_params(&mut rng);

    // Use stochastic gradient descent (Sgd), with a learning rate of 1e-2, and 0.9 momentum.
    let mut opt: Sgd<MLP> = Sgd::new(SgdConfig {
        lr: 1e-2,
        momentum: Some(Momentum::Classic(0.9)),
        weight_decay: None,
    });

    return (mlp, opt)
}

fn step(model: &mut MLP, opt: &mut Sgd<MLP>, y: Tensor1D<4, OwnedTape>, y_true: Tensor1D<4>) {
    // compute cross entropy loss
    let loss: Tensor0D<OwnedTape> = cross_entropy_with_logits_loss(y, y_true);
    println!("Loss={:?}", loss);
    // call `backward()` to compute gradients. The tensor *must* have `OwnedTape`!
    let gradients: Gradients = loss.backward();

    // pass the gradients & the model into the optimizer's update method
    opt.update(model, gradients);
}

fn predict_emittance_and_density(mlp: &MLP, views: &Vec<[f32; 3]>, points: &Vec<[f32; 3]>) -> Vec<Tensor1D<4, OwnedTape>> {
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

fn accumulate_radiance(predictions: Vec<Tensor1D<4, OwnedTape>>) ->  Vec<Tensor1D<3, OwnedTape>> {
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

const NUM_EPOCHS: usize = 5;

fn main() {
    let (views, points) = sample_points_along_view_directions();

    views.iter().for_each(|it| {
        println!("vector {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    let (mut mlp, mut opt): (MLP, Sgd<MLP>) = init_mlp();



    for epoch in 0..NUM_EPOCHS {
        println!("----------------- EPOCH {} --------------------", epoch);
        let predictions = predict_emittance_and_density(&mlp, &views, &points);

        predictions.iter().for_each(|it| {
            println!("prediction {:?}", it);
        });
        for prediction in predictions {
            step(&mut mlp, &mut opt, prediction, tensor([1., 0., 0., 0.,]));
        }
    }

}

#[test]
fn ray_direction_within_fov() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("x={} y={}", x, y);
    let mut to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    to[1] = 0.; //align in FOV plane
    let angle = vec3_dot(to, AT);
    println!("Cos {} vs Fov {}", angle, <f32>::cos(FOV/2.));

    assert!(angle >= <f32>::cos(FOV/2.))
}

#[test]
fn points_sampled_lie_on_ray() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("{} {}", x, y);
    let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    let points = sample_points_along_ray(FROM, to);

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|&it| {
        println!("^to {}", vec3_dot(vec3_normalized(it), to));
    });

    points.iter().for_each(|&it| {
        println!("|-to| {}", vec3_len(vec3_sub(vec3_normalized(it), to)));
    });

    assert!(points.iter().all(|&p| vec3_len(vec3_sub(vec3_normalized(p), to)) < 1e-6));
}

#[test]
fn points_sampled_ordered_by_t() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("{} {}", x, y);
    let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    let points = sample_points_along_ray(FROM, to);

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|&it| {
        println!("len {}", vec3_len(it));
    });

    let locations = points.iter().map(|&it| vec3_len(it)).collect::<Vec<f32>>();
    assert!((0..locations.len() - 1).all(|i| locations[i] <= locations[i + 1]));
}
