mod ray_sampling;
use model_tch::TchModel;
use ray_sampling::{HEIGHT, T_FAR, WIDTH};

mod image_loading;

mod input_transforms;

mod model_dfdx;
use model_dfdx::{from_u8_rgb, prediction_array_as_u32, rgba_to_u8_array};

mod model_tch;

mod display;
use display::run_window;

use textplots::{Chart, Plot, Shape};

use std::time::SystemTime;

use tensorboard_rs::summary_writer::SummaryWriter;

mod cli;
use clap::Parser;
use cli::Cli;

use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use tch::Tensor;

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

    let img_paths = image_loading::get_image_paths(args.img_dir, 0, 360, 10); // maybe restrict to 180 to avoid possible ray bugs
    let mut imgs = image_loading::load_multiple_images_as_arrays(img_paths); // TODO: split into training and held out views

    let mut model = model_tch::TchModel::new();
    if args.load_path != "" {
        model.load(&args.load_path);
    }

    let mut iter = 0;
    let mut writer = SummaryWriter::new(&format!("{}/{}", &args.log_dir, ts));

    log_params(&mut writer, &cli::get_scalars_as_map());

    let mut batch_losses: Vec<f32> = Vec::new();

    // training step takes place inside a window update callback
    // it used to be such that window was updated with each batch predictions but it takes way too long to draw on each iter
    let mut backbuffer = [0; WIDTH * HEIGHT];
    let update_window_buffer = |buffer: &mut Vec<u32>| {
        //predict emittance and density
        // let mut rng = thread_rng(); // Get a thread-local random number generator
        // imgs.shuffle(&mut rng);

        let (indices, query_points, distances, gold) = get_batch(&imgs, iter);
        let query_points_copy = query_points.clone();

        if iter % args.logging_steps == 0 {
            log_query_points(&mut writer, &query_points, &distances, iter);
        }

        let (predictions, densities) = model.predict(query_points, distances);

        if iter % args.logging_steps == 0 {
            log_density_maps(
                &mut writer,
                &query_points_copy,
                model_tch::get_predictions_as_array_vec(&densities),
                iter,
            );

            log_densities(
                &mut writer,
                &query_points_copy,
                model_tch::get_predictions_as_array_vec(&densities),
                iter,
            );
        }

        if args.do_train {
            let loss: f32 = model.step(&predictions, gold, &iter, args.accumulation_steps);

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
                draw_train_predictions(&mut backbuffer, indices, predictions, iter, &mut writer);
            } else {
                let angle = (iter as f32 / 180.) * std::f32::consts::PI; // / 2. + std::f32::consts::PI / 4.;
                draw_valid_predictions(
                    &mut backbuffer,
                    iter,
                    angle % (2. * std::f32::consts::PI),
                    &model,
                );
            }
        }

        draw_to_screen(buffer, &backbuffer, args.DEBUG, &imgs, &iter); // this is needed on each re-draw otherwise screen gets blank

        iter = iter + 1;
        if iter > args.num_iter {
            panic!("Reached maximum iterations")
        }
    };

    run_window(update_window_buffer, WIDTH, HEIGHT);
}

fn log_params(writer: &mut SummaryWriter, params: &HashMap<String, f32>) {
    for (key, value) in params {
        writer.add_scalar(key, *value, 0);
    }
}

fn log_query_points(
    writer: &mut SummaryWriter,
    query_points: &Vec<Vec<[f32; 3]>>,
    distances: &Vec<[f32; model_tch::NUM_POINTS]>,
    iter: usize,
) {
    let mut bucket_counts_y: [f64; 2000 as usize] = [0.; 2000 as usize];
    let mut bucket_counts_x: [f64; 2000 as usize] = [0.; 2000 as usize];
    let mut bucket_counts_z: [f64; 2000 as usize] = [0.; 2000 as usize];
    let mut bucket_counts_t: [f64; 2000 as usize] = [0.; 2000 as usize];

    for ray_points in query_points {
        for [world_x, world_y, world_z] in ray_points {
            bucket_counts_y[f32::floor(500. * (world_y + 1.)) as usize] += 1.;
            bucket_counts_x[f32::floor(500. * (world_x + 1.)) as usize] += 1.;
            bucket_counts_z[f32::floor(500. * (world_z + 1.)) as usize] += 1.;
        }
    }

    for ray_distances in distances {
        for t in ray_distances {
            bucket_counts_t[f32::floor(500. * t) as usize] += 1.;
        }
    }

    log_as_hist(writer, "world_y", bucket_counts_y, iter);
    log_as_hist(writer, "world_x", bucket_counts_x, iter);
    log_as_hist(writer, "world_z", bucket_counts_z, iter);
    log_as_hist(writer, "t", bucket_counts_t, iter);
}

fn log_densities(
    writer: &mut SummaryWriter,
    query_points: &Vec<Vec<[f32; 3]>>,
    densities: Vec<Vec<f32>>,
    iter: usize,
) {
    let mut bucket_counts_y: [f64; 2000 as usize] = [0.; 2000 as usize];
    let mut bucket_counts_x: [f64; 2000 as usize] = [0.; 2000 as usize];
    let mut bucket_counts_z: [f64; 2000 as usize] = [0.; 2000 as usize];

    for (ray_points, ray_densities) in query_points.iter().zip(densities) {
        for ([world_x, world_y, world_z], density) in ray_points.iter().zip(ray_densities) {
            let y = f32::floor(500. * (world_y + 1.)) as usize;
            let x = f32::floor(500. * (world_x + 1.)) as usize;
            let z = f32::floor(500. * (world_z + 1.)) as usize;

            bucket_counts_y[y] += density as f64;
            bucket_counts_x[x] += density as f64;
            bucket_counts_z[z] += density as f64;
        }
    }

    log_as_hist(writer, "density_y", bucket_counts_y, iter);
    log_as_hist(writer, "density_x", bucket_counts_x, iter);
    log_as_hist(writer, "density_z", bucket_counts_z, iter);
}

fn log_density_maps(
    writer: &mut SummaryWriter,
    query_points: &Vec<Vec<[f32; 3]>>,
    densities: Vec<Vec<f32>>,
    iter: usize,
) {
    let backbuffer_yx: &mut [u32; 100 * 100] = &mut [0u32; 100 * 100];
    let backbuffer_zx: &mut [u32; 100 * 100] = &mut [0u32; 100 * 100];
    let backbuffer_yz: &mut [u32; 100 * 100] = &mut [0u32; 100 * 100];
    for (ray_points, ray_densities) in query_points.iter().zip(densities) {
        for ([world_x, world_y, world_z], density) in ray_points.iter().zip(ray_densities) {
            let y = f32::floor(50. * (world_y + 1.)) as usize;
            let x = f32::floor(50. * (world_x + 1.)) as usize;
            let z = f32::floor(25. * (world_z + 1.)) as usize;

            let density_clamped = density.max(0.);

            backbuffer_yx[y * 100 + x] =
                prediction_array_as_u32(&[density_clamped, density_clamped, density_clamped, 1.]);

            backbuffer_zx[z * 100 + x] =
                prediction_array_as_u32(&[density_clamped, density_clamped, density_clamped, 1.]);

            backbuffer_yz[y * 100 + z] =
                prediction_array_as_u32(&[density_clamped, density_clamped, density_clamped, 1.]);
        }
    }

    writer.add_image(
        "density_yx",
        &backbuffer_yx
            .iter()
            .map(rgba_to_u8_array)
            .flatten()
            .collect::<Vec<u8>>(),
        &vec![3, 100, 100][..],
        iter,
    );
    writer.add_image(
        "density_zx",
        &backbuffer_zx
            .iter()
            .map(rgba_to_u8_array)
            .flatten()
            .collect::<Vec<u8>>(),
        &vec![3, 100, 100][..],
        iter,
    );

    writer.add_image(
        "density_yz",
        &backbuffer_yz
            .iter()
            .map(rgba_to_u8_array)
            .flatten()
            .collect::<Vec<u8>>(),
        &vec![3, 100, 100][..],
        iter,
    );
}

fn get_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
    iter: usize,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; model_tch::INDIM]>>,
    Vec<[f32; model_tch::NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    let n = iter % imgs.len();
    let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;

    let (indices, views, points) = ray_sampling::sample_points_tensor_along_view_directions(
        model_tch::NUM_RAYS,
        model_tch::NUM_POINTS,
        angle,
    ); // need mix rays from multiple views

    let gold: Vec<[f32; 4]> = indices
        .iter()
        .map(|[y, x]| imgs[n][y * WIDTH + x])
        .collect();

    //        let screen_coords: Vec<[f32; model_tch::INDIM]> = indices
    //            .iter()
    //            .map(input_transforms::scale_by_screen_size_and_fourier::<3>)
    //            .collect();

    let query_points: Vec<Vec<[f32; 3]>> = points
        .iter()
        .map(|ray_points| {
            ray_points
                .into_iter()
                .map(|([x, y, z], _)| [*x, *y, *z])
                .collect::<Vec<[f32; 3]>>()
        })
        .collect();

    let distances = points
        .into_iter()
        .map(|ray_points| {
            ray_points
                .into_iter()
                .map(|(_, t)| t)
                .collect::<Vec<f32>>()
                .try_into()
                .unwrap()
        })
        .collect();

    (indices, query_points, distances, gold)
}

fn draw_valid_predictions(
    backbuffer: &mut [u32; WIDTH * HEIGHT],
    iter: usize,
    angle: f32,
    model: &TchModel,
) {
    let mut indices: Vec<[usize; 2]> = Vec::new();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            indices.push([y as usize, x as usize])
        }
    }

    for batch_index in (0..indices.len() / model_tch::NUM_RAYS) {
        println!(
            "evaluating batch {:?} iter {:?} angle {:?} - {:?} out of {:?}",
            batch_index * model_tch::NUM_RAYS,
            iter,
            angle,
            (batch_index + 1) * model_tch::NUM_RAYS,
            indices.len()
        );
        let indices_batch: Vec<[usize; 2]> = indices
            [batch_index * model_tch::NUM_RAYS..(batch_index + 1) * model_tch::NUM_RAYS]
            .try_into()
            .unwrap();

        let points = ray_sampling::sample_points_tensor_for_rays(
            indices_batch,
            model_tch::NUM_POINTS,
            angle,
        );

        let query_points: Vec<Vec<[f32; model_tch::INDIM]>> = points
            .iter()
            .map(|ray_points| {
                ray_points
                    .into_iter()
                    .map(|([x, y, z], _)| [*x, *y, *z])
                    .collect::<Vec<[f32; 3]>>()
            })
            .collect();

        let distances: Vec<[f32; model_tch::NUM_POINTS]> = points
            .into_iter()
            .map(|ray_points| {
                ray_points
                    .into_iter()
                    .map(|(_, t)| t)
                    .collect::<Vec<f32>>()
                    .try_into()
                    .unwrap()
            })
            .collect();

        let (predictions, _) = model.predict(query_points, distances);

        for
                ([y, x], prediction) //[world_x, world_y, world_z]
            in indices[batch_index * model_tch::NUM_RAYS..(batch_index + 1) * model_tch::NUM_RAYS]
                .iter()
                .zip(model_tch::get_predictions_as_array_vec(&predictions).into_iter())
                // .zip(points)
                .into_iter()
            {
                backbuffer[y * WIDTH + x] = prediction_array_as_u32(&[prediction[0], prediction[1], prediction[2], 1.]);
            }
    }
}

fn draw_train_predictions(
    backbuffer: &mut [u32; WIDTH * HEIGHT],
    indices: Vec<[usize; 2]>,
    predictions: Tensor,
    iter: usize,
    writer: &mut SummaryWriter,
) {
    let mut bucket_counts_sy: [f64; HEIGHT] = [0.; HEIGHT];
    let mut bucket_counts_sx: [f64; WIDTH] = [0.; WIDTH];

    // let mut bucket_counts_r: [f64; WIDTH as usize] = [0.; WIDTH as usize];
    // let mut bucket_counts_g: [f64; T_FAR as usize] = [0.; T_FAR as usize];
    // let mut bucket_counts_b: [f64; T_FAR as usize] = [0.; T_FAR as usize];

    // write batch predictions to backbuffer to display until next eval
    for
            ([y, x], prediction) //[world_x, world_y, world_z]
        in indices
            .iter()
            .zip(model_tch::get_predictions_as_array_vec(&predictions).into_iter())
            // .zip(points)
            .into_iter()
        {
            backbuffer[y * WIDTH + x] = prediction_array_as_u32(&[prediction[0], prediction[1], prediction[2], 1.]);
            bucket_counts_sy[*y] += 1.;
            bucket_counts_sx[*x] += 1.;

            // if *y == HEIGHT - 1 {
            //     bucket_counts_r[f32::floor(*x as f32) as usize] = prediction[0] as f64;
            // }
            // bucket_counts_g[f32::floor(world_z) as usize] += prediction[1] as f64;
            // bucket_counts_b[f32::floor(world_z) as usize] += prediction[2] as f64;
        }

    log_as_hist(writer, "screen_y", bucket_counts_sy, iter);
    log_as_hist(writer, "screen_x", bucket_counts_sx, iter);

    // log_as_hist(writer, "density_r", bucket_counts_r, iter);
    // log_as_hist(writer, "density_g", bucket_counts_g, iter);
    // log_as_hist(writer, "density_b", bucket_counts_b, iter);

    writer.add_image(
        "prediction",
        &backbuffer
            .iter()
            .map(rgba_to_u8_array)
            .flatten()
            .collect::<Vec<u8>>(),
        &vec![3, WIDTH, HEIGHT][..],
        iter,
    ); //TODO probably also save gold view
}

fn log_as_hist<const RANGE: usize>(
    writer: &mut SummaryWriter,
    tag: &str,
    bucket_counts: [f64; RANGE],
    iter: usize,
) {
    writer.add_histogram_raw(
        tag,
        0 as f64,
        RANGE as f64,
        bucket_counts.iter().sum::<f64>() as f64,
        bucket_counts.iter().sum::<f64>() as f64,
        bucket_counts.iter().map(|o| o * o).sum::<f64>() as f64,
        &(1..=RANGE as i64).map(|o| o as f64).collect::<Vec<f64>>(),
        &bucket_counts,
        iter,
    );
}

fn draw_to_screen(
    buffer: &mut Vec<u32>,
    backbuffer: &[u32; WIDTH * HEIGHT], //img: &Vec<[f32; 4]>
    DEBUG: bool,
    imgs: &Vec<Vec<[f32; 4]>>,
    iter: &usize,
) {
    // draw from either backbuffer or gold image
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if DEBUG {
                let gold = imgs[iter % imgs.len()][y * WIDTH + x];
                buffer[y * WIDTH + x] = from_u8_rgb(
                    (gold[0] * 255.) as u8,
                    (gold[1] * 255.) as u8,
                    (gold[2] * 255.) as u8,
                );
            } else {
                buffer[y * WIDTH + x] = backbuffer[y * WIDTH + x];
            }
        }
    }
}
