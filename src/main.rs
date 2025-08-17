mod cli;
use clap::Parser;
use cli::Cli;
mod image_loading;
use image_loading::*;
mod ray_sampling;
use ray_sampling::*;
mod dataset;
use dataset::*;
mod input_transforms;
mod model;
use model::{
    tensor_from_2d, tensor_from_3d, tensor_to_2d, NeRF, BATCH_SIZE, INDIM, LABELS, NUM_POINTS,
    NUM_RAYS,
};
mod display;
use display::*;
mod logging;
use logging::*;
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

    let img_paths = get_image_paths(args.img_dir, args.view_start, args.view_end, args.view_step);
    let imgs = image_loading::load_multiple_images_as_arrays(img_paths.clone());
    let view_angles = image_loading::get_view_angles(args.num_views_per_hemisphere);

    let mut model = model::NeRF::new(); // let vs = nn::VarStore::new(Device::Mps);// let mut model = model::DensityNet::new(&vs.root());
    let mut trainer = model::Trainer::new(&model.vs, args.learning_rate); // let mut trainer = model::Trainer::new(&vs, args.learning_rate);
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
        let (indices, query_points, distances, gold) = get_multiview_batch(&imgs, &view_angles); //get_train_batch(&imgs, iter); // // // //get_sphere_train_batch(angle);// let (indices, query_points, gold) = get_density_batch(&imgs, iter);
        let (colors, densities) = model.predict(
            tensor_from_3d(&query_points),
            tensor_from_2d::<NUM_POINTS>(&distances),
        ); // let densities = model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points));

        if iter % args.logging_steps == 0 {
            log_screen_coords(&mut writer, &indices, iter);
            log_query_points_as_maps(&mut writer, &query_points, iter);
            log_query_distances(&mut writer, &distances, iter);
            // log_density_maps(&mut writer, &query_points, tensor_to_2d(&densities), iter);
        }
        panic!("");

        if args.do_train {
            let loss = trainer.step(&colors, tensor_from_2d::<{ LABELS as usize }>(&gold), &iter); // let loss = trainer.step(&densities, tensor_from_3d::<{ LABELS as usize }>(&gold), &iter);
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
                draw_predictions(&mut backbuffer, &indices, colors);
                log_prediction(&mut writer, &mut backbuffer, iter);
            } else {
                // draw_valid_predictions(&mut backbuffer, iter, &model);
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

// check points sampled from different view rays get same density estimates
// fn measure_view_invariance(
//     writer: &mut SummaryWriter,
//     model: &DensityNet,
//     iter: usize,
//     angle: f32,
// ) {
//     writer.add_scalar(
//         "density0",
//         model
//             .predict::<1, 1, 1>(tensor_from_3d(&vec![vec![[0., 0., 0.]]]))
//             .get(0)
//             .get(0)
//             .try_into()
//             .unwrap(),
//         iter,
//     );
//     print!("sampling {:?} rays for angles 0. and {:?}", NUM_RAYS, angle);

//     let mut indices: Vec<[usize; 2]> = Vec::new();

//     for y in 0..HEIGHT {
//         for x in 0..WIDTH {
//             indices.push([y as usize, x as usize])
//         }
//     }

//     // let mut indices1 = get_random_screen_coords(NUM_RAYS);
//     // let mut indices2 = get_random_screen_coords(NUM_RAYS);
//     let rays1 = sample_and_rotate_rays_for_screen_coords(&indices, 0.);
//     let rays2 = sample_and_rotate_rays_for_screen_coords(&indices, angle);

//     let (rays1_intersections, rays2_intersections, rays1_keep, rays2_keep) =
//         get_view_rays_intersections(rays1, rays2, angle);

//     let len_intersections1 = rays1_intersections
//         .iter()
//         .flatten()
//         .collect::<Vec<&[f32; 3]>>()
//         .len();

//     let len_intersections2 = rays2_intersections
//         .iter()
//         .flatten()
//         .collect::<Vec<&[f32; 3]>>()
//         .len();

//     println!(
//         " -> {:?} and {:?} intersections",
//         len_intersections1, len_intersections2,
//     );

//     if len_intersections1 > 0 && len_intersections2 > 0 {
//         //only check query points on rays having intersections
//         // let mut rays1_keep_iter = rays1_keep.iter();
//         // indices1.retain(|_| *rays1_keep_iter.next().unwrap());
//         // let mut rays2_keep_iter = rays2_keep.iter();
//         // indices2.retain(|_| *rays2_keep_iter.next().unwrap());
//         // println!(
//         //     "indices retained {:?} and {:?}",
//         //     indices1.len(),
//         //     indices2.len()
//         // );

//         let (query_points1, _) =
//             sample_points_for_screen_coords_and_view_angles(&indices, NUM_POINTS, 0., false);
//         let (query_points2, _) =
//             sample_points_for_screen_coords_and_view_angles(&indices, NUM_POINTS, angle, false);

//         let densities1 =
//             model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points1));

//         let densities2 =
//             model.predict::<BATCH_SIZE, NUM_RAYS, NUM_POINTS>(tensor_from_3d(&query_points2));

//         let query_points_densities_intersected1: Vec<([f32; 3], f32)> = query_points1
//             .iter()
//             .zip(tensor_to_2d(&densities1))
//             .zip(rays1_intersections)
//             .map(|((points, densities), intersections)| {
//                 points
//                     .clone()
//                     .into_iter()
//                     .zip(densities)
//                     .filter(move |(p, d)| intersections.iter().any(|i| dist(*p, *i) < TOL))
//             })
//             .flatten()
//             .collect();

//         let query_points_densities_intersected2: Vec<([f32; 3], f32)> = query_points2
//             .iter()
//             .zip(tensor_to_2d(&densities2))
//             .zip(rays2_intersections)
//             .map(|((points, densities), intersections)| {
//                 points
//                     .clone()
//                     .into_iter()
//                     .zip(densities)
//                     .filter(move |(p, d)| intersections.iter().any(|i| dist(*p, *i) < TOL))
//             })
//             .flatten()
//             .collect();

//         let mut consistency_error = 0.;
//         let mut query_point_pairs: Vec<([f32; 3], [f32; 3])> = Vec::new();
//         for (qp1, d1) in query_points_densities_intersected1.iter() {
//             for (qp2, d2) in query_points_densities_intersected2.iter() {
//                 if dist(*qp1, *qp2) < TOL {
//                     query_point_pairs.push((*qp1, *qp2));
//                     consistency_error += (d1 - d2).abs();
//                 }
//             }
//         }
//         consistency_error /= query_point_pairs.len() as f32;
//         println!(
//             "intersected points {:?} and {:?} -> {:?} ({:?} err)",
//             query_points_densities_intersected1.len(),
//             query_points_densities_intersected2.len(),
//             query_point_pairs.len(),
//             consistency_error
//         );
//         writer.add_scalar("consistency_error", consistency_error, iter);

//         log_rays_intersections(
//             writer,
//             query_point_pairs.into_iter().map(|(a, b)| a).collect(),
//             iter,
//         );
//     }
// }

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
