//We synthesize images by sampling 5D coordinates (location and viewing direction) along camera rays (a),
//feeding those locations into an MLP to produce a color and volume density (b),
//  We encourage the representation to be multiview consistent by restricting the network to predict the volume density σ as a function of only the location x,
//  while allowing the RGB color c to be predicted as a function of both location and viewing direction.
//  To accomplish this, the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer),
//  and outputs σ and a 256-dimensional feature vector. This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional
//  fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.
//and using volume ren- dering techniques to composite these values into an image (c
//TODO: curious how it would do without relying on volume rendering approximation and density
mod ray_sampling;
use ray_sampling::{HEIGHT, WIDTH};

mod image_loading;

use std;
use std::convert::TryInto;

mod mlp;
use mlp::Mlp;

mod model;
use model::{MLP, Sgd, Adam, init_mlp, predict_emittance_and_density, prediction_as_u32, prediction_array_as_u32, step, tensor, from_u8_rgb, HasArrayData};

mod display;
use display::run_window;

use textplots::{Chart, Plot, Shape};

const DEBUG: bool = false;
// TODO:
/*
- Reproduce with CocoNet
- debug polar and corner coordinates
- 32x32 image or increase model size
- learn shaded render
- try reproduce horse
- sample highest error samples
*/
fn main() {
    let img = image_loading::load_image_as_array("spheres/image-0-basic-128.png");
    println!("image {:?} pixels", img.len());
    let mut backbuffer = [0; WIDTH * HEIGHT];

    let (mut mlp, mut opt): (MLP, Adam<MLP>) = init_mlp();
    let mut batch_losses: Vec<f32> = Vec::new();

    let mut update_window_buffer = |buffer: &mut Vec<u32>| {
        if !DEBUG {
            let (indices, views, points) = ray_sampling::sample_points_batch_along_view_directions(model::BATCH_SIZE);
//            let (indices, views, points) = ray_sampling::sample_points_along_view_directions();
            let screen_coords: Vec<[f32; 2]> = indices.iter().map(|&e| [e[0] as f32 / HEIGHT as f32, e[1] as f32 / WIDTH as f32]).map(|e| [
                e[0] - 0.5,
                e[1] - 0.5//,
//                (1. - e[0]),
//                (1. - e[1]),
//                f32::sqrt((-(e[0] - 0.5)) * (-(e[0] - 0.5)) + (e[1] - 0.5) * (e[1] - 0.5)),
//                1./ (f32::tan((-(e[0] - 0.5)) / (e[1] - 0.5 + 1e-6) + 1e-6))
            ]).collect();
            let gold: Vec<[f32; 4]> = indices.iter().map(|[y, x]| img[y * WIDTH + x]).collect();

            let predictions = predict_emittance_and_density(&mlp, screen_coords, views, points);

            for ([y, x], prediction) in indices[..model::BATCH_SIZE].iter().zip(predictions.data().into_iter()) {
                backbuffer[y * WIDTH + x] = prediction_array_as_u32(prediction);
            }

            let loss: f32 = step(&mut mlp, &mut opt, predictions, gold[..model::BATCH_SIZE].to_vec());

            println!("avg loss={:.4}", loss);
            batch_losses.push(loss);
            Chart::new(120, 40, 0., batch_losses.len() as f32)
            .lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
            .display();

            if (batch_losses.len() * model::BATCH_SIZE) % (img.len() * 10) == 0 {
                backbuffer = [0; WIDTH * HEIGHT];
            }
        }

        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if DEBUG {
                    let gold = img[y * WIDTH + x];
                    buffer[y * WIDTH + x] = from_u8_rgb((gold[0] * 255.) as u8, (gold[1]*255.) as u8, (gold[2]*255.) as u8);
                }
                else{
                    buffer[y * WIDTH + x] = backbuffer[y * WIDTH + x];
                }
            }
        }

    };
    run_window(update_window_buffer, WIDTH, HEIGHT);
}
