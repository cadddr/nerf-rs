mod cli;
use clap::Parser;
use cli::Cli;
mod image_loading;
mod ray_sampling;
use ray_sampling::*;
mod input_transforms;
mod model;
use model::{
    tensor_from_2d, tensor_from_3d, tensor_to_array_vec, NeRF, BATCH_SIZE, INDIM, LABELS,
    NUM_POINTS, NUM_RAYS,
};
use tch::{kind, nn, nn::Optimizer, nn::OptimizerConfig, Device, Kind, Tensor};
use vecmath::*;
mod display;
use display::*;
mod logging;
use logging::*;
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::time::SystemTime;
use tensorboard_rs::summary_writer::SummaryWriter;
use textplots::{Chart, Plot, Shape};

use crate::model::DensityNet;

fn main() {
    /*
    Main loop. Reads image(s), inits model, runs train loop (within window refresh handler);
    on eval - draw to backbuffer, which is displayed on every frame
    */
    let ts = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let args = Cli::parse();
    let mut writer = SummaryWriter::new(&format!("{}/{}", &args.log_dir, ts));
    log_params(&mut writer, &cli::get_scalars_as_map());

    let img_paths = image_loading::get_image_paths(
        args.img_dir,
        args.view_start_h,
        args.view_end_h,
        args.view_step_h,
    );
    let imgs = image_loading::load_multiple_images_as_arrays(img_paths);

    // let mut model = model::NeRF::new();
    let vs = nn::VarStore::new(Device::Mps);
    let mut model = model::DensityNet::new(&vs.root());
    // let mut trainer = model::Trainer::new(&model.vs, args.learning_rate);
    let mut trainer = model::Trainer::new(&vs, args.learning_rate);
    log_params(&mut writer, &model::hparams());

    // if args.load_path != "" {
    //     model.load(&format!("{}/{}", args.save_dir, &args.load_path));
    // }

    // training step takes place inside a window update callback
    // it used to be such that window was updated with each batch predictions but it takes way too long to draw on each iter
    let mut iter = 0;
    let mut batch_losses: Vec<f32> = Vec::new();
    let mut backbuffer = [0; WIDTH * HEIGHT];
    let update_window_buffer = |buffer: &mut Vec<u32>| {
        // let (indices, query_points, distances, gold) = get_train_batch(&imgs, iter);
        let (query_points, gold) = get_density_batch(&imgs, iter);

        // let (predictions, densities) = model.predict(
        //     tensor_from_3d(&query_points),
        //     tensor_from_2d::<NUM_POINTS>(&distances),
        // );
        let densities =
            model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points));

        if iter % args.logging_steps == 0 {
            // log_screen_coords(&mut writer, &indices, iter);
            // log_query_points(&mut writer, &query_points, &distances, iter);
            log_density_maps(
                &mut writer,
                &query_points,
                tensor_to_array_vec(&densities),
                iter,
            );
            log_densities(
                &mut writer,
                &query_points,
                tensor_to_array_vec(&densities),
                iter,
            );
        }

        if args.do_train {
            // let loss: f32 = trainer.step(
            //     &predictions,
            //     tensor_from_2d::<{ LABELS as usize }>(&gold),
            //     &iter,
            //     args.accumulation_steps,
            // );
            let loss: f32 = trainer.step(
                &densities,
                tensor_from_3d::<{ LABELS as usize }>(&gold),
                &iter,
                args.accumulation_steps,
            );
            println!("iter={}, loss={:.16}", iter, loss);
            writer.add_scalar("loss", loss, iter);

            batch_losses.push(loss);
            Chart::new(120, 40, 0., batch_losses.len() as f32)
                .lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
                .display();

            // if iter % args.save_steps == 0 {
            //     model.save(&format!("{}/checkpoint-{}-{}.ot", args.save_dir, ts, iter));
            // }
        }
        if iter % args.eval_steps == 0 {
            backbuffer = [0; WIDTH * HEIGHT];
            if args.eval_on_train {
                // draw_predictions(&mut backbuffer, indices, predictions);
                // log_prediction(&mut writer, &mut backbuffer, iter);
            } else {
                // draw_valid_predictions(&mut backbuffer, iter, &model);
            }
            let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
            let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;
            measure_view_invariance(&mut writer, &model, iter, angle);
        }
        draw_to_screen(buffer, &backbuffer, args.debug, &imgs, &iter); // this is needed on each re-draw otherwise screen gets blank

        iter = iter + 1;
        if iter > args.num_iter {
            panic!("Reached maximum iterations")
        }
    };

    run_window(update_window_buffer, WIDTH, HEIGHT);
}

#[test]
fn display_ray_intersections() {
    let update_window_buffer = |buffer: &mut Vec<u32>| {
        for y in 0..HEIGHT {
            for x in 0..WIDTH {
                if trace_ray_intersections(x as f32, y as f32) {
                    buffer[y * WIDTH + x] = display::from_u8_rgb(255u8, 255u8, 255u8);
                } else {
                    buffer[y * WIDTH + x] = display::from_u8_rgb(0u8, 0u8, 0u8);
                }
            }
        }
    };
    run_window(update_window_buffer, WIDTH, HEIGHT);
}

fn measure_view_invariance(
    writer: &mut SummaryWriter,
    model: &DensityNet,
    iter: usize,
    angle: f32,
) {
    // writer.add_scalar(
    //     "density0",
    //     model
    //         .predict::<1, 1, 1>(tensor_from_3d(&vec![vec![[0., 0., 0.]]]))
    //         .get(0)
    //         .get(0)
    //         .try_into()
    //         .unwrap(),
    //     iter,
    // );
    print!("sampling {:?} rays for angles 0. and {:?}", NUM_RAYS, angle);
    let rays1 = sample_and_rotate_rays_for_screen_coords(&get_random_screen_coords(NUM_RAYS), 0.);
    let rays2 =
        sample_and_rotate_rays_for_screen_coords(&get_random_screen_coords(NUM_RAYS), angle);

    let intersections = get_view_rays_intersections(rays1, rays2, angle);
    println!("-> {:?} intersections", intersections.len());
    log_rays_intersections(writer, intersections, iter);
}

fn get_random_screen_coords(num_rays: usize) -> Vec<[usize; 2]> {
    // generating random tensors is faster
    let coord_y: Vec<i64> = Vec::try_from(Tensor::randint(
        HEIGHT as i64,
        &[num_rays as i64],
        kind::FLOAT_CPU,
    ))
    .unwrap();

    let coord_x: Vec<i64> = Vec::try_from(Tensor::randint(
        WIDTH as i64,
        &[num_rays as i64],
        kind::FLOAT_CPU,
    ))
    .unwrap();

    let indices: Vec<[usize; 2]> = coord_y
        .iter()
        .zip(coord_x.iter())
        .map(|(y, x)| [*y as usize, *x as usize])
        .collect();

    return indices;
}

fn get_density_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
    iter: usize,
) -> (Vec<Vec<[f32; INDIM as usize]>>, Vec<Vec<[f32; 1]>>) {
    let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
    let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;
    let indices = get_random_screen_coords(NUM_RAYS);
    let (query_points, _) =
        sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle); // need mix rays from multiple views
    let mut gold: Vec<Vec<[f32; 1]>> = Vec::new();

    for (ray_points) in &query_points {
        let mut ray_gold: Vec<[f32; 1]> = Vec::new();
        for [x, y, z] in ray_points {
            let distance_from_center = (x * x + y * y + z * z).sqrt();
            let true_density = if distance_from_center < 0.5 { 1.0 } else { 0.0 };
            ray_gold.push([true_density]);
        }
        gold.push(ray_gold);
    }

    return (query_points, gold);
}

// gets query points for random screen coords and views for training
fn get_train_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
    iter: usize,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; INDIM as usize]>>,
    Vec<[f32; NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    // let mut rng = thread_rng(); // Get a thread-local random number generator
    // imgs.shuffle(&mut rng);
    //
    let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
    let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;

    let indices = get_random_screen_coords(NUM_RAYS);
    //        let screen_coords: Vec<[f32; model_tch::INDIM]> = indices
    //            .iter()
    //            .map(input_transforms::scale_by_screen_size_and_fourier::<3>)
    //            .collect();

    let (query_points, distances) =
        sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle); // need mix rays from multiple views

    let gold: Vec<[f32; 4]> = indices
        .iter()
        .map(|[y, x]| imgs[n][y * WIDTH + x])
        .collect();

    (indices, query_points, distances, gold)
}

// queries model for batches of all screen coordinates and draws to backbuffer
fn draw_valid_predictions(backbuffer: &mut [u32; WIDTH * HEIGHT], iter: usize, model: &NeRF) {
    let mut indices: Vec<[usize; 2]> = Vec::new();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            indices.push([y as usize, x as usize])
        }
    }

    let mut angle = (iter as f32 / 180.) * std::f32::consts::PI;
    angle %= 2. * std::f32::consts::PI;

    for batch_index in (0..indices.len() / NUM_RAYS) {
        println!(
            "evaluating batch {:?} iter {:?} angle {:?} - {:?} out of {:?}",
            batch_index * NUM_RAYS,
            iter,
            angle,
            (batch_index + 1) * NUM_RAYS,
            indices.len()
        );
        let indices_batch: Vec<[usize; 2]> = indices
            [batch_index * NUM_RAYS..(batch_index + 1) * NUM_RAYS]
            .try_into()
            .unwrap();

        let (query_points, distances) =
            sample_and_rotate_ray_points_for_screen_coords(&indices_batch, NUM_POINTS, angle);

        let (predictions, _) = model.predict(
            tensor_from_3d(&query_points),
            tensor_from_2d::<NUM_POINTS>(&distances),
        );
        draw_predictions(backbuffer, indices_batch, predictions);
    }
}

fn draw_predictions(
    backbuffer: &mut [u32; WIDTH * HEIGHT],
    indices: Vec<[usize; 2]>,
    predictions: Tensor,
) {
    // write batch predictions to backbuffer to display until next eval
    for ([y, x], prediction) in indices
        .iter()
        .zip(tensor_to_array_vec(&predictions).into_iter())
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
