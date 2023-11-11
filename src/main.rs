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

mod input_transforms;

mod model_dfdx;
use model_dfdx::{prediction_as_u32, prediction_array_as_u32, from_u8_rgb, rgba_to_u8_array};

mod model_tch;

mod display;
use display::run_window;

use textplots::{Chart, Plot, Shape};

use clap::Parser;

use std::fs::File;
use std::io::{Write, BufReader, BufRead, Error};

use tensorboard_rs::summary_writer::SummaryWriter;

const DEBUG: bool = false;
const REFRESH_EPOCHS: usize = 100;
const num_iter: usize = 50000;
const eval_steps: usize = 200;
// TODO:
/*
- sample highest error samples
- predict shading as different channels
*/

#[derive(Parser)]
struct Cli {
	#[arg(default_value = "spheres/image-0.png")]
    img_path: String,
	
	#[arg(default_value = "./logdir")]
    log_dir: String,
	
	#[arg(default_value = "checkpoint.ot")]
	save_path: String,
	
	#[arg(default_value = "")]
	load_path: String
}

fn main() {
	/*
	Main loop. Reads image(s), inits model, runs train loop (within window refresh handler);
	on eval - draw to backbuffer, which is displayed on every frame
	*/
	
	let args = Cli::parse();
	let mut writer = SummaryWriter::new(&args.log_dir);
	
    let img = image_loading::load_image_as_array(&args.img_path);
    println!("image {:?} pixels", img.len());
    let mut backbuffer = [0; WIDTH * HEIGHT];

	// let mut model = model_dfdx::DfdxMlp::new();
	let mut model = model_tch::TchModel::new();
	if args.load_path != "" {
		model.load(&args.load_path);
	}

	let mut iter = 0;
	let mut batch_losses: Vec<f32> = Vec::new();
    let mut update_window_buffer = |buffer: &mut Vec<u32>| {
        if !DEBUG {
			//predict emittance and density
			let (indices, views, points) = ray_sampling::sample_points_tensor_along_view_directions(model.BATCH_SIZE());
			let gold: Vec<[f32; 4]> = indices.iter().map(|[y, x]| img[y * WIDTH + x]).collect();
			let screen_coords: Vec<[f32; model_tch::INDIM]> = indices.iter().map(input_transforms::scale_by_screen_size_and_fourier::<3>).collect();
			
			let predictions = model.predict(points.clone(), views.clone(), screen_coords.clone());
			
			if (iter % eval_steps == 0) {
				// refresh backbuffer every few steps
	            if (batch_losses.len() * model.BATCH_SIZE()) % (img.len() * REFRESH_EPOCHS) == 0 {
	                backbuffer = [0; WIDTH * HEIGHT];
	            }
				
				// write batch predictions to backbuffer to display until next eval
				for ([y, x], prediction) in indices[..model.BATCH_SIZE()].iter().zip(model.get_predictions_as_array_vec(&predictions).into_iter()) {
                	backbuffer[y * WIDTH + x] = prediction_array_as_u32(&prediction);
            	}
				
				writer.add_image("prediction", &backbuffer.iter().map(rgba_to_u8_array).flatten().collect::<Vec<u8>>(), &vec![3, WIDTH, HEIGHT][..], iter);
				model.save(&args.save_path);
			}

			let loss: f32 = model.step(predictions, gold[..model.BATCH_SIZE()].to_vec());
			// loss & plotting
            println!("iter={}, loss={:.16}", iter, loss);
			writer.add_scalar("loss", loss, iter);
			
			batch_losses.push(loss);
            Chart::new(120, 40, 0., batch_losses.len() as f32)
            	.lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
            	.display();
        }
		
		draw_to_screen(buffer, &backbuffer, &img);
		iter = iter + 1;
		if iter > num_iter {
			panic!("Reached maximum iterations")
		}
    };
    run_window(update_window_buffer, WIDTH, HEIGHT);
}

fn draw_to_screen(buffer: &mut Vec<u32>, backbuffer: &[u32; WIDTH * HEIGHT], img: &Vec<[f32; 4]>) {
	// draw from either backbuffer or gold image
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
}
