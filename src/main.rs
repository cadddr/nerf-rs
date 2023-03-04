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
use model::{MLP, Sgd, init_mlp, predict_emittance_and_density, prediction_as_u32, step, tensor, from_u8_rgb, HasArrayData};

mod display;
use display::run_window;

const NUM_EPOCHS: usize = 1000;

fn main() {
    let img = image_loading::load_image_as_array("spheres/image-0.png");
    println!("image {:?} pixels", img.len());

    let (mut mlp, mut opt): (MLP, Sgd<MLP>) = init_mlp();
    let (indices, views, points) = ray_sampling::sample_points_along_view_directions();

    let mut update_window_buffer = |buffer: &mut Vec<u32>| {
        let predictions = predict_emittance_and_density(&mlp, &views, &points);
        let num_predictions = predictions.len();
        let mut losses: Vec<f32> = Vec::new();
        for ((y, x), prediction) in indices.iter().zip(predictions.into_iter()) {
            let gold = img[y * WIDTH + x];
            if (num_predictions <= 10) {
                print!("pixel {}, {} ", y, x);
                print!("gold {{{:.2} {:.2} {:.2} {:.2}}} vs ", gold[0], gold[1], gold[2], gold[3]);
                println!("pred {{{:.2} {:.2} {:.2} {:.2}}} ", prediction.data()[0], prediction.data()[1], prediction.data()[2], prediction.data()[3]);
            }

//            buffer[y * WIDTH + x] = from_u8_rgb((gold[0] * 255.) as u8, (gold[1] * 255.) as u8, (gold[2] * 255.) as u8);
            buffer[y * WIDTH + x] = prediction_as_u32(&prediction);

            let loss: f32 = step(&mut mlp, &mut opt, prediction, tensor(gold));
            losses.push(loss);
        }

        println!("avg loss={:.4}", losses.iter().sum::<f32>() as f32 / losses.len() as f32);
    };
    run_window(update_window_buffer, WIDTH, HEIGHT);
}

//fn main() {
//    let img = image_loading::load_image_as_array("spheres/image-0.png");
//    println!("image {:?}", img.len());
//
//    let mut update_window_buffer = |buffer: &mut Vec<u32>| {
//        for y in 0..HEIGHT {
//            for x in 0..WIDTH {
//                let gold = img[y * WIDTH + x];
//                buffer[y * WIDTH + x] = from_u8_rgb((gold[0] * 255.) as u8, (gold[1]*255.) as u8, (gold[2]*255.) as u8);
//            }
//        }
//    };
//
//    run_window(update_window_buffer, WIDTH, HEIGHT);
//}

