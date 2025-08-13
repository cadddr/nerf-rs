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
use std::{f32, time::SystemTime};
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

    let mut model = model::NeRF::new();
    // let vs = nn::VarStore::new(Device::Mps);
    // let mut model = model::DensityNet::new(&vs.root());
    let mut trainer = model::Trainer::new(&model.vs, args.learning_rate);
    // let mut trainer = model::Trainer::new(&vs, args.learning_rate);
    log_params(&mut writer, &model::hparams());

    if args.load_path != "" {
        model.load(&format!("{}/{}", args.save_dir, &args.load_path));
    }

    // training step takes place inside a window update callback
    // it used to be such that window was updated with each batch predictions but it takes way too long to draw on each iter
    let mut iter = 0;
    let mut batch_losses: Vec<f32> = Vec::new();
    let mut backbuffer = [0; WIDTH * HEIGHT];
    let update_window_buffer = |buffer: &mut Vec<u32>| {
        // let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
        // let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;
        let (indices, query_points, distances, gold) = get_multiview_batch(&imgs); //get_sphere_train_batch(angle);
                                                                                   // let (indices, query_points, gold) = get_density_batch(&imgs, iter);

        let (predictions, densities) = model.predict(
            tensor_from_3d(&query_points),
            tensor_from_2d::<NUM_POINTS>(&distances),
        );
        // let densities =
        //     model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points));

        if iter % args.logging_steps == 0 {
            log_screen_coords(&mut writer, &indices, iter);
            log_query_points(&mut writer, &query_points, iter);
            log_query_distances(&mut writer, &distances, iter);
            log_density_maps(
                &mut writer,
                &query_points,
                tensor_to_array_vec(&densities),
                iter,
            );
        }

        if args.do_train {
            let loss: f32 = trainer.step(
                &predictions,
                tensor_from_2d::<{ LABELS as usize }>(&gold),
                &iter,
                args.accumulation_steps,
            );
            // let loss: f32 = trainer.step(
            //     &densities,
            //     tensor_from_3d::<{ LABELS as usize }>(&gold),
            //     &iter,
            //     args.accumulation_steps,
            // );
            println!("iter={}, loss={:.16}", iter, loss);
            writer.add_scalar("loss", loss, iter);

            batch_losses.push(loss);
            Chart::new(120, 40, 0., batch_losses.len() as f32)
                .lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
                .display();

            if iter % args.save_steps == 0 {
                model.save(&format!("{}/checkpoint-{}-{}.ot", args.save_dir, ts, iter));
            }
        }
        if iter % args.eval_steps == 0 {
            backbuffer = [0; WIDTH * HEIGHT];
            if args.eval_on_train {
                draw_predictions(&mut backbuffer, &indices, predictions);
                log_prediction(&mut writer, &mut backbuffer, iter);
            } else {
                draw_valid_predictions(&mut backbuffer, iter, &model);
            }
            // let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
            // let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;
            // if angle > 0. {
            //     measure_view_invariance(&mut writer, &model.density, iter, angle);
            // }
        }
        if args.debug {
            backbuffer = [0; WIDTH * HEIGHT];
            draw_predictions(
                &mut backbuffer,
                &indices,
                tensor_from_2d::<{ LABELS as usize }>(&gold).view((NUM_RAYS as i64, LABELS)),
            );
        }

        draw_to_screen(buffer, &backbuffer); // this is needed on each re-draw otherwise screen gets blank

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

// check points sampled from different view rays get same density estimates
fn measure_view_invariance(
    writer: &mut SummaryWriter,
    model: &DensityNet,
    iter: usize,
    angle: f32,
) {
    writer.add_scalar(
        "density0",
        model
            .predict::<1, 1, 1>(tensor_from_3d(&vec![vec![[0., 0., 0.]]]))
            .get(0)
            .get(0)
            .try_into()
            .unwrap(),
        iter,
    );
    print!("sampling {:?} rays for angles 0. and {:?}", NUM_RAYS, angle);

    let mut indices: Vec<[usize; 2]> = Vec::new();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            indices.push([y as usize, x as usize])
        }
    }

    // let mut indices1 = get_random_screen_coords(NUM_RAYS);
    // let mut indices2 = get_random_screen_coords(NUM_RAYS);
    let rays1 = sample_and_rotate_rays_for_screen_coords(&indices, 0.);
    let rays2 = sample_and_rotate_rays_for_screen_coords(&indices, angle);

    let (rays1_intersections, rays2_intersections, rays1_keep, rays2_keep) =
        get_view_rays_intersections(rays1, rays2, angle);

    let len_intersections1 = rays1_intersections
        .iter()
        .flatten()
        .collect::<Vec<&[f32; 3]>>()
        .len();

    let len_intersections2 = rays2_intersections
        .iter()
        .flatten()
        .collect::<Vec<&[f32; 3]>>()
        .len();

    println!(
        " -> {:?} and {:?} intersections",
        len_intersections1, len_intersections2,
    );

    if len_intersections1 > 0 && len_intersections2 > 0 {
        //only check query points on rays having intersections
        // let mut rays1_keep_iter = rays1_keep.iter();
        // indices1.retain(|_| *rays1_keep_iter.next().unwrap());
        // let mut rays2_keep_iter = rays2_keep.iter();
        // indices2.retain(|_| *rays2_keep_iter.next().unwrap());
        // println!(
        //     "indices retained {:?} and {:?}",
        //     indices1.len(),
        //     indices2.len()
        // );

        let (query_points1, _) =
            sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, 0., false);
        let (query_points2, _) =
            sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, false);

        let densities1 =
            model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points1));

        let densities2 =
            model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points2));

        let query_points_densities_intersected1: Vec<([f32; 3], f32)> = query_points1
            .iter()
            .zip(tensor_to_array_vec(&densities1))
            .zip(rays1_intersections)
            .map(|((points, densities), intersections)| {
                points
                    .clone()
                    .into_iter()
                    .zip(densities)
                    .filter(move |(p, d)| intersections.iter().any(|i| dist(*p, *i) < TOL))
            })
            .flatten()
            .collect();

        let query_points_densities_intersected2: Vec<([f32; 3], f32)> = query_points2
            .iter()
            .zip(tensor_to_array_vec(&densities2))
            .zip(rays2_intersections)
            .map(|((points, densities), intersections)| {
                points
                    .clone()
                    .into_iter()
                    .zip(densities)
                    .filter(move |(p, d)| intersections.iter().any(|i| dist(*p, *i) < TOL))
            })
            .flatten()
            .collect();

        let mut consistency_error = 0.;
        let mut query_point_pairs: Vec<([f32; 3], [f32; 3])> = Vec::new();
        for (qp1, d1) in query_points_densities_intersected1.iter() {
            for (qp2, d2) in query_points_densities_intersected2.iter() {
                if dist(*qp1, *qp2) < TOL {
                    query_point_pairs.push((*qp1, *qp2));
                    consistency_error += (d1 - d2).abs();
                }
            }
        }
        consistency_error /= query_point_pairs.len() as f32;
        println!(
            "intersected points {:?} and {:?} -> {:?} ({:?} err)",
            query_points_densities_intersected1.len(),
            query_points_densities_intersected2.len(),
            query_point_pairs.len(),
            consistency_error
        );
        writer.add_scalar("consistency_error", consistency_error, iter);

        log_rays_intersections(
            writer,
            query_point_pairs.into_iter().map(|(a, b)| a).collect(),
            iter,
        );
    }
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

fn get_sphere_density_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
    iter: usize,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; INDIM as usize]>>,
    Vec<Vec<[f32; 1]>>,
) {
    let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
    let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;
    let indices = get_random_screen_coords(NUM_RAYS);
    let (query_points, _) =
        sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, true); // need mix rays from multiple views
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

    return (indices, query_points, gold);
}

fn get_multiview_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; INDIM as usize]>>,
    Vec<[f32; NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    let indices = get_random_screen_coords(NUM_RAYS);
    let bsz = NUM_RAYS / imgs.len();

    let mut query_points: Vec<Vec<[f32; INDIM as usize]>> = Vec::new();
    let mut distances: Vec<[f32; NUM_POINTS]> = Vec::new();
    let mut gold: Vec<[f32; 4]> = Vec::new();

    let image_indices: Vec<i64> = Vec::try_from(Tensor::randint(
        imgs.len() as i64,
        &[imgs.len() as i64],
        kind::FLOAT_CPU,
    ))
    .unwrap();

    for (i, n) in image_indices.iter().enumerate() {
        // println!("view {:?} of {:?}", n, image_indices.len());
        let angle = (*n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI; // all views encompass 360 degrees
        let indices_batch: Vec<[usize; 2]> = indices[i as usize * bsz..(i as usize + 1) * bsz]
            .try_into()
            .unwrap();

        let (query_points_batch, distances_batch) =
            sample_and_rotate_ray_points_for_screen_coords(&indices_batch, NUM_POINTS, angle, true);

        let gold_batch: Vec<[f32; 4]> = indices_batch
            .iter()
            .map(|[y, x]| imgs[*n as usize][y * WIDTH + x])
            .collect();
        //
        // let (_, query_points_batch, distances_batch, gold_batch) =
        //     get_sphere_train_batch(indices_batch, angle);

        query_points.extend(query_points_batch);
        distances.extend(distances_batch);
        gold.extend(gold_batch);
    }

    return (indices, query_points, distances, gold);
}

fn get_sphere_train_batch(
    indices: Vec<[usize; 2]>,
    angle: f32,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; INDIM as usize]>>,
    Vec<[f32; NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    // let indices = get_random_screen_coords(NUM_RAYS);
    let mut gold: Vec<[f32; 4]> = Vec::new();
    for [y, x] in indices.clone() {
        let y_ = y as f32 / HEIGHT as f32;
        let x_ = x as f32 / WIDTH as f32;

        let distance_from_center = ((x_ - 0.5) * (x_ - 0.5) + (y_ - 0.5) * (y_ - 0.5)).sqrt();
        let distance_from_center_left =
            ((x_ - 0.25) * (x_ - 0.25) + (y_ - 0.5) * (y_ - 0.5)).sqrt();
        let distance_from_center_right =
            ((x_ - 0.75) * (x_ - 0.75) + (y_ - 0.5) * (y_ - 0.5)).sqrt();

        // if angle == 0. {
        //     gold.push([1., 0., 0., 1.]);
        // } else if f32::abs(angle - f32::consts::FRAC_PI_2) < TOL {
        //     gold.push([0., 1., 0., 1.]);
        // } else if (f32::abs(angle - 3. * f32::consts::FRAC_PI_2) < TOL) {
        //     gold.push([0., 0., 1., 1.]);
        // } else {
        //     gold.push([1., 1., 1., 1.]);
        // }
        // continue;

        if distance_from_center < 0.25
            || (f32::abs(angle - f32::consts::FRAC_PI_2) < TOL
                || f32::abs(angle - 3. * f32::consts::FRAC_PI_2) < TOL)
                && (distance_from_center_left < 0.125 || distance_from_center_right < 0.125)
        {
            gold.push([1., 1., 1., 1.]);
        } else {
            gold.push([0., 0., 0., 0.]);
        }
    }

    let (query_points, distances) =
        sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, true);

    return (indices, query_points, distances, gold);
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
        sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, true); // need mix rays from multiple views

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
            sample_and_rotate_ray_points_for_screen_coords(&indices_batch, NUM_POINTS, angle, true);

        let (predictions, _) = model.predict(
            tensor_from_3d(&query_points),
            tensor_from_2d::<NUM_POINTS>(&distances),
        );
        draw_predictions(backbuffer, &indices_batch, predictions);
    }
}

fn draw_predictions(
    backbuffer: &mut [u32; WIDTH * HEIGHT],
    indices: &Vec<[usize; 2]>,
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
