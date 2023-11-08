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
use model_dfdx::{prediction_as_u32, prediction_array_as_u32, from_u8_rgb};

mod model_tch;

mod display;
use display::run_window;

use textplots::{Chart, Plot, Shape};

const DEBUG: bool = false;
const REFRESH_EPOCHS: usize = 5;
// TODO:
/*
- Reproduce with CocoNet
- debug polar and corner coordinates
- 32x32 image or increase model size
- learn shaded render
- try reproduce horse
- sample highest error samples
- predict shading as different channels
*/

// batch 128 steps 1k
// tch 
// nerf	23.6	2:00.51	8	0.6	6.68	34945	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)
// dfdx 
// nerf	10.7	54.83	6	0.0	0.09	35105	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	


// batch 256 steps 512
// tch
// nerf	35.9	1:36.08	9	0.7	5.45	37060	pavlik	Yes	0 bytes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)
// dfdx
// nerf	17.7	46.25	7	0.0	0.04	35328	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)

// batch 512 steps 256
// tch 300 steps
// nerf	59.6	1:28.92	9	0.8	5.35	37298	pavlik	Yes	0 bytes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)
// tch 512 steps
// nerf	76.5	2:51.25	9	0.6	9.93	37298	pavlik	Yes	0 bytes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)
// dfdx
// nerf	28.6	39.13	8	0.0	0.02	35511	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)

// batch 1024 steps 512 /4
// tch
// nerf	99.3	5:19.31	9	0.0	18.96	43796	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)
// dfdx
// nerf	53.6	2:12.63	8	0.0	0.04	35684	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)

// batch 2048 steps 256 /4
// tch
// nerf	89.8	4:36.62	8	0.0	18.29	43978	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	
// dfdx
// nerf	97.9	2:07.93	6	0.0	0.02	35879	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)

// batch 4096 steps 256 /8
// tch
// nerf	97.3	8:59.30	10	0.0	28.05	44307	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	
// dfdx
// nerf	100.5	4:01.19	7	0.0	0.02	36167	pavlik	Yes	140.0 MB	273	0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)

// batch 8k steps 128 /8
// tch
// nerf	100.9	10:12.12	9	0.0	26.30	44815	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	
// dfdx
// nerf	100.0	4:11.52	6	0.0	0.01	36345	pavlik	Yes	186.0 MB	229	0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	

// batch 16k steps 32 /4
// tch
// nerf	99.7	5:21.61	13	0.0	13.11	45035	pavlik	Yes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	

// dfdx
// nerf	100.1	2:05.91	7	0.0	0.00	36564	pavlik	Yes	0 bytes		0 bytes	0 bytes	0 bytes	Intel	No	No	No	(null)	(null)	No	0 bytes	0	0 bytes	0	0 bytes	0 bytes	-	0 bytes	0 bytes	No	(null)	


fn main() {
	let num_iter = 50000;
    let img = image_loading::load_image_as_array("spheres/image-0.png");//image-0-basic-128.png");
    println!("image {:?} pixels", img.len());
    let mut backbuffer = [0; WIDTH * HEIGHT];

	// let mut model = model_dfdx::DfdxMlp::new();
	let mut model = model_tch::TchModel::new();

    let mut batch_losses: Vec<f32> = Vec::new();
	
    let (indices, views, points) =
        ray_sampling::sample_points_batch_along_view_directions(model.BATCH_SIZE());
	let gold: Vec<[f32; 4]> = indices.iter().map(|[y, x]| img[y * WIDTH + x]).collect();
	
    let screen_coords: Vec<[f32; model_tch::INDIM]> = indices.iter().map(input_transforms::scale_by_screen_size_and_center).collect();

    // let mut update_window_buffer = |buffer: &mut Vec<u32>| {
	for iter in 0..num_iter {
        if !DEBUG {
            
			
			//predict emittance and density
			let predictions = model.predict(screen_coords.clone(), views.clone(), points.clone());
			
			if (iter % 100 == 0) {
				for ([y, x], prediction) in indices[..model.BATCH_SIZE()].iter().zip(model.get_predictions_as_array_vec(&predictions).into_iter()) {
                	backbuffer[y * WIDTH + x] = prediction_array_as_u32(&prediction);
            	}
			}

			let loss: f32 = model.step(predictions, gold[..model.BATCH_SIZE()].to_vec());
			batch_losses.push(loss);
			
			// loss & plotting
            println!("avg loss={:.16}", loss);
            Chart::new(120, 40, 0., batch_losses.len() as f32)
            	.lineplot(&Shape::Continuous(Box::new(|x| batch_losses[x as usize])))
            	.display();

			// refresh backbuffer every few steps
            if (batch_losses.len() * model.BATCH_SIZE()) % (img.len() * REFRESH_EPOCHS) == 0 {
                backbuffer = [0; WIDTH * HEIGHT];
            }
        }

		// draw from either backbuffer or gold image
       // for y in 0..H// EIGHT {
//             for x in 0..WIDTH {
//                 if DEBUG {
//                     let gold = img[y * WIDTH + x];
//                     buffer[y * WIDTH + x] = from_u8_rgb((gold[0] * 255.) as u8, (gold[1]*255.) as u8, (gold[2]*255.) as u8);
//                 }
//                 else{
//                     buffer[y * WIDTH + x] = backbuffer[y * WIDTH + x];
//                 }
//             }
//         }

    };
    // run_window(update_window_buffer, WIDTH, HEIGHT);
}
