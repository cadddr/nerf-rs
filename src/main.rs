mod ray_sampling;
use model::NeRF;
use ray_sampling::{HEIGHT, WIDTH};
mod display;
mod image_loading;
mod input_transforms;
mod model;
use display::{draw_to_screen, prediction_array_as_u32, rgba_to_u8_array, run_window};
use std::time::SystemTime;
use tensorboard_rs::summary_writer::SummaryWriter;
use textplots::{Chart, Plot, Shape};
mod cli;
use clap::Parser;
use cli::Cli;
use rand::seq::SliceRandom;
use rand::thread_rng;
use tch::Tensor;
mod logging;
use logging::*;

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
    let mut trainer = model::Trainer::new(&model.vs, args.learning_rate);
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
        let (indices, query_points, distances, gold) = get_batch(&imgs, iter);
        let (predictions, densities) = model.predict(&query_points, &distances);

        if iter % args.logging_steps == 0 {
            log_query_points(&mut writer, &query_points, &distances, iter);

            log_density_maps(
                &mut writer,
                &query_points,
                model::get_predictions_as_array_vec(&densities),
                iter,
            );

            log_densities(
                &mut writer,
                &query_points,
                model::get_predictions_as_array_vec(&densities),
                iter,
            );
        }

        if args.do_train {
            let loss: f32 = trainer.step(&predictions, &gold, &iter, args.accumulation_steps);
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
                draw_valid_predictions(&mut backbuffer, iter, &model);
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

fn get_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
    iter: usize,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; model::INDIM]>>,
    Vec<[f32; model::NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    // let mut rng = thread_rng(); // Get a thread-local random number generator
    // imgs.shuffle(&mut rng);
    //
    let n = iter % imgs.len();
    let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;

    let (indices, _, query_points, distances) =
        ray_sampling::sample_camera_rays_points_and_distances(
            model::NUM_RAYS,
            model::NUM_POINTS,
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

    (indices, query_points, distances, gold)
}

fn draw_valid_predictions(backbuffer: &mut [u32; WIDTH * HEIGHT], iter: usize, model: &NeRF) {
    let mut indices: Vec<[usize; 2]> = Vec::new();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            indices.push([y as usize, x as usize])
        }
    }

    let mut angle = (iter as f32 / 180.) * std::f32::consts::PI;
    angle %= 2. * std::f32::consts::PI;

    for batch_index in (0..indices.len() / model::NUM_RAYS) {
        println!(
            "evaluating batch {:?} iter {:?} angle {:?} - {:?} out of {:?}",
            batch_index * model::NUM_RAYS,
            iter,
            angle,
            (batch_index + 1) * model::NUM_RAYS,
            indices.len()
        );
        let indices_batch: Vec<[usize; 2]> = indices
            [batch_index * model::NUM_RAYS..(batch_index + 1) * model::NUM_RAYS]
            .try_into()
            .unwrap();

        let (query_points, distances) =
            ray_sampling::sample_ray_points_and_distances_for_screen_coords(
                &indices_batch,
                model::NUM_POINTS,
                angle,
            );

        let (predictions, _) = model.predict(&query_points, &distances);

        for ([y, x], prediction) in indices
            [batch_index * model::NUM_RAYS..(batch_index + 1) * model::NUM_RAYS]
            .iter()
            .zip(model::get_predictions_as_array_vec(&predictions).into_iter())
            .into_iter()
        {
            backbuffer[y * WIDTH + x] =
                prediction_array_as_u32(&[prediction[0], prediction[1], prediction[2], 1.]);
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

    // write batch predictions to backbuffer to display until next eval
    for ([y, x], prediction) in indices
        .iter()
        .zip(model::get_predictions_as_array_vec(&predictions).into_iter())
        .into_iter()
    {
        backbuffer[y * WIDTH + x] =
            prediction_array_as_u32(&[prediction[0], prediction[1], prediction[2], 1.]);
        bucket_counts_sy[*y] += 1.;
        bucket_counts_sx[*x] += 1.;
    }

    log_as_hist(writer, "screen_y", bucket_counts_sy, iter);
    log_as_hist(writer, "screen_x", bucket_counts_sx, iter);

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
