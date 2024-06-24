mod ray_sampling;
use ray_sampling::{HEIGHT, T_FAR, WIDTH};

mod image_loading;

mod input_transforms;

mod model_dfdx;
use model_dfdx::{prediction_array_as_u32, rgba_to_u8_array};

mod model_tch;

mod display;
use display::run_window;

use textplots::{Chart, Plot, Shape};

use std::time::SystemTime;

use tensorboard_rs::summary_writer::SummaryWriter;

mod cli;
use clap::Parser;
use cli::Cli;

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

    let img_paths = image_loading::get_image_paths(args.img_dir, 0, 360, 40);
    let imgs = image_loading::load_multiple_images_as_arrays(img_paths); // TODO: split into training and held out views

    let mut model = model_tch::TchModel::new();
    if args.load_path != "" {
        model.load(&args.load_path);
    }

    let mut iter = 0;
    let mut writer = SummaryWriter::new(&format!("{}/{}", &args.log_dir, ts));

    let mut batch_losses: Vec<f32> = Vec::new();

    // training step takes place inside a window update callback
    // it used to be such that window was updated with each batch predictions but it takes way too long to draw on each iter
    let mut backbuffer = [0; WIDTH * HEIGHT];
    let update_window_buffer = |buffer: &mut Vec<u32>| {
        //predict emittance and density
        let (indices, query_points, distances, gold) = get_batch(&imgs, iter);
        let predictions = model.predict(query_points, distances);
        let loss: f32 = model.step(&predictions, gold);

        println!("iter={}, loss={:.16}", iter, loss);
        writer.add_scalar("loss", loss, iter);

        batch_losses.push(loss);

        Chart::new(120, 40, 0., batch_losses.len() as f32)
            .lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
            .display();

        if iter % args.eval_steps == 0 {
            backbuffer = [0; WIDTH * HEIGHT];
            draw_train_predictions(&mut backbuffer, indices, predictions, iter, &mut writer);

            model.save(&format!("{}/checkpoint-{}-{}.ot", args.save_dir, ts, iter));
        }

        draw_to_screen(buffer, &backbuffer, args.DEBUG); // this is needed on each re-draw otherwise screen gets blank

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
    Vec<Vec<[f32; model_tch::INDIM]>>,
    Vec<[f32; model_tch::NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    let n = iter % imgs.len();
    let angle = (n as f32 / imgs.len() as f32) * std::f32::consts::PI / 2.;

    let (indices, views, points) = ray_sampling::sample_points_tensor_along_view_directions(
        model_tch::NUM_RAYS,
        model_tch::NUM_POINTS,
        angle,
    );

    let gold: Vec<[f32; 4]> = indices
        .iter()
        .map(|[y, x]| imgs[n][y * WIDTH + x])
        .collect();

    //        let screen_coords: Vec<[f32; model_tch::INDIM]> = indices
    //            .iter()
    //            .map(input_transforms::scale_by_screen_size_and_fourier::<3>)
    //            .collect();

    let query_points = points
        .iter()
        .map(|ray_points| {
            ray_points
                .into_iter()
                .map(|([x, y, z], _)| [*x, *y, *z, angle])
                .collect::<Vec<[f32; 4]>>()
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

fn draw_train_predictions(
    backbuffer: &mut [u32; WIDTH * HEIGHT],
    indices: Vec<[usize; 2]>,
    predictions: Tensor,
    iter: usize,
    writer: &mut SummaryWriter,
) {
    let mut bucket_counts_sy: [f64; HEIGHT] = [0.; HEIGHT];
    let mut bucket_counts_sx: [f64; WIDTH] = [0.; WIDTH];
    let mut bucket_counts_y: [f64; 10000 as usize] = [0.; 10000 as usize];
    let mut bucket_counts_x: [f64; 10000 as usize] = [0.; 10000 as usize];
    let mut bucket_counts_z: [f64; T_FAR as usize] = [0.; T_FAR as usize];
    let mut bucket_counts_r: [f64; WIDTH as usize] = [0.; WIDTH as usize];
    let mut bucket_counts_g: [f64; T_FAR as usize] = [0.; T_FAR as usize];
    let mut bucket_counts_b: [f64; T_FAR as usize] = [0.; T_FAR as usize];

    // write batch predictions to backbuffer to display until next eval
    for
            ([y, x], prediction) //[world_x, world_y, world_z]
        in indices
            .iter()
            .zip(model_tch::get_predictions_as_array_vec(&predictions).into_iter())
            // .zip(points)
            .into_iter()
        {
            backbuffer[y * WIDTH + x] = prediction_array_as_u32(&prediction);
            bucket_counts_sy[*y] += 1.;
            bucket_counts_sx[*x] += 1.;
            // bucket_counts_y[f32::floor(1000. * world_y) as usize] += 1.;
            // bucket_counts_x[f32::floor(1000. * world_x) as usize] += 1.;
            // bucket_counts_z[f32::floor(world_z) as usize] += 1.;
            if *y == HEIGHT - 1 {
                bucket_counts_r[f32::floor(*x as f32) as usize] = prediction[0] as f64;
            }
            // bucket_counts_g[f32::floor(world_z) as usize] += prediction[1] as f64;
            // bucket_counts_b[f32::floor(world_z) as usize] += prediction[2] as f64;
        }

    log_as_hist(writer, "screen_y", bucket_counts_sy, iter);
    log_as_hist(writer, "screen_x", bucket_counts_sx, iter);
    log_as_hist(writer, "world_y", bucket_counts_y, iter);
    log_as_hist(writer, "world_x", bucket_counts_x, iter);
    log_as_hist(writer, "world_z", bucket_counts_z, iter);
    log_as_hist(writer, "density_r", bucket_counts_r, iter);
    log_as_hist(writer, "density_g", bucket_counts_g, iter);
    log_as_hist(writer, "density_b", bucket_counts_b, iter);

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
) {
    // draw from either backbuffer or gold image
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if DEBUG {
                //                let gold = img[y * WIDTH + x];
                //                buffer[y * WIDTH + x] = from_u8_rgb(
                //                    (gold[0] * 255.) as u8,
                //                    (gold[1] * 255.) as u8,
                //                    (gold[2] * 255.) as u8,
                //                );
            } else {
                buffer[y * WIDTH + x] = backbuffer[y * WIDTH + x];
            }
        }
    }
}
