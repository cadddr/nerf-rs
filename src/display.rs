extern crate minifb;
use minifb::{Key, Window, WindowOptions};
use tch::Tensor;

use crate::{
    model::{tensor_from_2d, tensor_from_3d, tensor_to_2d, NeRF, NUM_POINTS, NUM_RAYS},
    ray_sampling::*,
};

pub fn run_window<F: FnMut(&mut Vec<u32>)>(
    mut update_window_buffer: F,
    width: usize,
    height: usize,
) {
    let mut window =
        Window::new("NeRF Live Preview", width, height, WindowOptions::default()).unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_micros(1000)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut buffer: Vec<u32> = vec![0; width * height];
        update_window_buffer(&mut buffer);
        window.update_with_buffer(&buffer, width, height).unwrap();
    }
}

pub fn draw_to_screen(
    buffer: &mut Vec<u32>,
    backbuffer: &[u32; WIDTH * HEIGHT], //img: &Vec<[f32; 4]>
) {
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            buffer[y * WIDTH + x] = backbuffer[y * WIDTH + x];
        }
    }
}

pub fn from_u8_rgb(r: u8, g: u8, b: u8) -> u32 {
    let (r, g, b) = (r as u32, g as u32, b as u32);
    (r << 16) | (g << 8) | b
}

pub fn rgba_to_u8_array(rgba: &u32) -> [u8; 3] {
    [(*rgba >> 16) as u8, (*rgba >> 8) as u8, *rgba as u8]
}

pub fn prediction_array_as_u32(rgba: &[f32; 4]) -> u32 {
    return from_u8_rgb(
        (rgba[0] * 255.) as u8,
        (rgba[1] * 255.) as u8,
        (rgba[2] * 255.) as u8,
    );
}

// queries model for batches of all screen coordinates and draws to backbuffer
// pub fn draw_valid_predictions(backbuffer: &mut [u32; WIDTH * HEIGHT], iter: usize, model: &NeRF) {
//     let mut indices: Vec<[usize; 2]> = Vec::new();

//     for y in 0..HEIGHT {
//         for x in 0..WIDTH {
//             indices.push([y as usize, x as usize])
//         }
//     }

//     let mut angle = (iter as f32 / 180.) * std::f32::consts::PI;
//     angle %= 2. * std::f32::consts::PI;

//     for batch_index in (0..indices.len() / NUM_RAYS) {
//         println!(
//             "evaluating batch {:?} iter {:?} angle {:?} - {:?} out of {:?}",
//             batch_index * NUM_RAYS,
//             iter,
//             angle,
//             (batch_index + 1) * NUM_RAYS,
//             indices.len()
//         );
//         let indices_batch: Vec<[usize; 2]> = indices
//             [batch_index * NUM_RAYS..(batch_index + 1) * NUM_RAYS]
//             .try_into()
//             .unwrap();

//         let (query_points, distances) = sample_points_for_screen_coords_and_view_angles(
//             &indices_batch,
//             NUM_POINTS,
//             angle,
//             true,
//         );

//         let (predictions, _) = model.predict(
//             tensor_from_3d(&query_points),
//             tensor_from_2d::<NUM_POINTS>(&distances),
//         );
//         draw_predictions(backbuffer, &indices_batch, predictions);
//     }
// }

pub fn draw_predictions(
    backbuffer: &mut [u32; WIDTH * HEIGHT],
    indices: &Vec<[usize; 2]>,
    predictions: Tensor,
) {
    // write batch predictions to backbuffer to display until next eval
    for ([y, x], prediction) in indices
        .iter()
        .zip(tensor_to_2d(&predictions).into_iter())
        .into_iter()
    {
        backbuffer[y * WIDTH + x] =
            prediction_array_as_u32(&[prediction[0], prediction[1], prediction[2], 1.]);
    }
}

// // draws training predictions to backbuffer and logs
// fn draw_train_predictions(
//     backbuffer: &mut [u32; WIDTH * HEIGHT],
//     indices: Vec<[usize; 2]>,
//     predictions: Tensor,
//     iter: usize,
//     writer: &mut SummaryWriter,
// ) {
//     // write batch predictions to backbuffer to display until next eval
//     for ([y, x], prediction) in indices
//         .iter()
//         .zip(get_predictions_as_array_vec(&predictions).into_iter())
//         .into_iter()
//     {
//         backbuffer[y * WIDTH + x] =
//             prediction_array_as_u32(&[prediction[0], prediction[1], prediction[2], 1.]);
//     }
// }
