//We synthesize images by sampling 5D coordinates (location and viewing direction) along camera rays (a),
//feeding those locations into an MLP to produce a color and volume density (b),
//  We encourage the representation to be multiview consistent by restricting the network to predict the volume density σ as a function of only the location x,
//  while allowing the RGB color c to be predicted as a function of both location and viewing direction.
//and using volume ren- dering techniques to composite these values into an image (c
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

use clap::Parser;

use std::time::SystemTime;

use tensorboard_rs::summary_writer::SummaryWriter;

const DEBUG: bool = false;

#[derive(Parser)]
struct Cli {
    #[arg(long, default_value = "spheres/image-0.png")]
    img_path: String,
    //    #[arg(long, default_value_t = vec!["spheres/image-0.png".to_string(), "spheres/image-89.png".to_string()])]
    //    img_paths: Vec<String>,
    #[arg(long, default_value = "logs")]
    log_dir: String,

    #[arg(long, default_value = "checkpoints")]
    save_dir: String,

    #[arg(long, default_value = "")]
    load_path: String,

    #[arg(long, default_value_t = 50000)]
    num_iter: usize,

    #[arg(long, default_value_t = 101)]
    eval_steps: usize,

    #[arg(long, default_value_t = 100)]
    refresh_epochs: usize,
}

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

    let imgs = image_loading::load_multiple_images_as_arrays(vec![
        "monkey-128-no-shading/image-0.png",
        "monkey-128-no-shading/image-40.png",
        "monkey-128-no-shading/image-80.png",
        "monkey-128-no-shading/image-120.png",
        "monkey-128-no-shading/image-240.png",
        "monkey-128-no-shading/image-359.png",
    ]); //TODO:

    let mut backbuffer = [0; WIDTH * HEIGHT];

    let mut model = model_tch::TchModel::new();
    if args.load_path != "" {
        model.load(&args.load_path);
    }

    let mut iter = 0;
    let mut writer = SummaryWriter::new(&format!("{}/{}", &args.log_dir, ts));

    let mut batch_losses: Vec<f32> = Vec::new();
    let update_window_buffer = |buffer: &mut Vec<u32>| {
        //predict emittance and density
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

        let predictions = model.predict(
            points
                .into_iter()
                .map(|ray_points| {
                    ray_points
                        .into_iter()
                        .map(|([x, y, z], _)| [x, y, z, angle])
                        .collect::<Vec<[f32; 4]>>()
                })
                .collect(),
            points
                .into_iter()
                .map(|ray_points| {
                    ray_points
                        .into_iter()
                        .map(|(_, t)| t)
                        .collect::<Vec<f32>>()
                        .try_into()
                        .unwrap()
                })
                .collect(),
        );

        // panic!("{:?}", predictions.get(0));

        if iter % args.eval_steps == 0 {
            // refresh backbuffer every few steps
            //            if (batch_losses.len() * model.BATCH_SIZE()) % (img.len() * args.refresh_epochs) == 0 {
            backbuffer = [0; WIDTH * HEIGHT];
            //            }

            let mut bucket_counts_sy: [f64; HEIGHT] = [0.; HEIGHT];
            let mut bucket_counts_sx: [f64; WIDTH] = [0.; WIDTH];
            let mut bucket_counts_y: [f64; 10000 as usize] = [0.; 10000 as usize];
            let mut bucket_counts_x: [f64; 10000 as usize] = [0.; 10000 as usize];
            let mut bucket_counts_z: [f64; T_FAR as usize] = [0.; T_FAR as usize];
            let mut bucket_counts_r: [f64; T_FAR as usize] = [0.; T_FAR as usize];
            let mut bucket_counts_g: [f64; T_FAR as usize] = [0.; T_FAR as usize];
            let mut bucket_counts_b: [f64; T_FAR as usize] = [0.; T_FAR as usize];

            // write batch predictions to backbuffer to display until next eval
            for
                ([y, x], prediction) //[world_x, world_y, world_z]
            in indices
                .iter()
                .zip(model.get_predictions_as_array_vec(&predictions).into_iter())
                // .zip(points)
                .into_iter()
            {
                backbuffer[y * WIDTH + x] = prediction_array_as_u32(&prediction);
                bucket_counts_sy[*y] += 1.;
                bucket_counts_sx[*x] += 1.;
                // bucket_counts_y[f32::floor(1000. * world_y) as usize] += 1.;
                // bucket_counts_x[f32::floor(1000. * world_x) as usize] += 1.;
                // bucket_counts_z[f32::floor(world_z) as usize] += 1.;

                // bucket_counts_r[f32::floor(world_z) as usize] += prediction[0] as f64;
                // bucket_counts_g[f32::floor(world_z) as usize] += prediction[1] as f64;
                // bucket_counts_b[f32::floor(world_z) as usize] += prediction[2] as f64;
            }

            writer.add_scalar("angle", 180. * angle / std::f32::consts::PI, iter);
            log_as_hist(&mut writer, "screen_y", bucket_counts_sy, iter);
            log_as_hist(&mut writer, "screen_x", bucket_counts_sx, iter);
            log_as_hist(&mut writer, "world_y", bucket_counts_y, iter);
            log_as_hist(&mut writer, "world_x", bucket_counts_x, iter);
            log_as_hist(&mut writer, "world_z", bucket_counts_z, iter);
            log_as_hist(&mut writer, "density_r", bucket_counts_r, iter);
            log_as_hist(&mut writer, "density_g", bucket_counts_g, iter);
            log_as_hist(&mut writer, "density_b", bucket_counts_b, iter);

            writer.add_image(
                "prediction",
                &backbuffer
                    .iter()
                    .map(rgba_to_u8_array)
                    .flatten()
                    .collect::<Vec<u8>>(),
                &vec![3, WIDTH, HEIGHT][..],
                iter,
            );
            model.save(&format!("{}/checkpoint-{}-{}.ot", args.save_dir, ts, iter));
        }

        let loss: f32 = model.step(predictions, gold);
        // loss & plotting
        println!(
            "iter={}, angle={:.4} loss={:.16}",
            iter,
            180. * angle / std::f32::consts::PI,
            loss
        );
        writer.add_scalar("loss", loss, iter);

        batch_losses.push(loss);
        Chart::new(120, 40, 0., batch_losses.len() as f32)
            .lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
            .display();

        draw_to_screen(buffer, &backbuffer); //, &img);

        iter = iter + 1;
        if iter > args.num_iter {
            panic!("Reached maximum iterations")
        }
    };

    run_window(update_window_buffer, WIDTH, HEIGHT);
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
