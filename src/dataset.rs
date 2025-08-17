use std::f32;

use tch::{kind, Tensor};

use crate::{
    model::{INDIM, NUM_POINTS, NUM_RAYS},
    ray_sampling::*,
};

pub fn get_random_screen_coords(num_rays: usize) -> Vec<[usize; 2]> {
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

// pub fn get_sphere_density_batch(
//     imgs: &Vec<Vec<[f32; 4]>>,
//     iter: usize,
// ) -> (
//     Vec<[usize; 2]>,
//     Vec<Vec<[f32; INDIM as usize]>>,
//     Vec<Vec<[f32; 1]>>,
// ) {
//     let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
//     let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;
//     let indices = get_random_screen_coords(NUM_RAYS);
//     let (query_points, _) =
//         sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, true); // need mix rays from multiple views
//     let mut gold: Vec<Vec<[f32; 1]>> = Vec::new();

//     for (ray_points) in &query_points {
//         let mut ray_gold: Vec<[f32; 1]> = Vec::new();
//         for [x, y, z] in ray_points {
//             let distance_from_center = (x * x + y * y + z * z).sqrt();
//             let true_density = if distance_from_center < 0.5 { 1.0 } else { 0.0 };
//             ray_gold.push([true_density]);
//         }
//         gold.push(ray_gold);
//     }

//     return (indices, query_points, gold);
// }

pub fn get_multiview_batch(
    imgs: &Vec<Vec<[f32; 4]>>,
    view_angles: &Vec<(f32, f32)>,
) -> (
    Vec<[usize; 2]>,
    Vec<Vec<[f32; INDIM as usize]>>,
    Vec<[f32; NUM_POINTS]>,
    Vec<[f32; 4]>,
) {
    let indices = get_random_screen_coords(NUM_RAYS);
    let extra = NUM_RAYS % imgs.len();
    assert_eq!(
        extra,
        0,
        "Can't divide {:?} rays evenly among {:?} views, got extra {:?}",
        NUM_RAYS,
        imgs.len(),
        extra
    );
    let bsz = NUM_RAYS / imgs.len();

    let mut query_points: Vec<Vec<[f32; INDIM as usize]>> = Vec::new();
    let mut distances: Vec<[f32; NUM_POINTS]> = Vec::new();
    let mut gold: Vec<[f32; 4]> = Vec::new();

    let view_index: Vec<i64> = Vec::try_from(Tensor::randint(
        imgs.len() as i64,
        &[imgs.len() as i64],
        kind::FLOAT_CPU,
    ))
    .unwrap();

    for (i, n) in view_index.iter().enumerate() {
        println!("view {:?} of {:?}", n, view_index.len());
        // let angle = (*n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI; // all views encompass 360 degrees
        let (yawAngle, pitchAngle) = view_angles[*n as usize];
        let indices_batch: Vec<[usize; 2]> = indices[i as usize * bsz..(i as usize + 1) * bsz]
            .try_into()
            .unwrap();

        let (query_points_batch, distances_batch) = sample_and_rotate_ray_points_for_screen_coords(
            &indices_batch,
            NUM_POINTS,
            yawAngle,
            pitchAngle,
            true,
        );

        let gold_batch: Vec<[f32; 4]> = indices_batch
            .iter()
            .map(|[y, x]| imgs[*n as usize][y * WIDTH + x])
            .collect();
        //
        // let (_, query_points_batch, distances_batch, gold_batch) =
        //     get_sphere_train_batch(indices_batch, angle);

        println!(
            "query_batch={:?} distances_batch={:?} gold_batch={:?}",
            query_points_batch.len(),
            distances_batch.len(),
            gold_batch.len()
        );
        query_points.extend(query_points_batch);
        distances.extend(distances_batch);
        gold.extend(gold_batch);
    }

    println!(
        "indices_total={:?} query_total={:?} distances_total={:?} gold_total={:?}",
        indices.len(),
        query_points.len(),
        distances.len(),
        gold.len()
    );

    return (indices, query_points, distances, gold);
}

// pub fn get_sphere_train_batch(
//     indices: Vec<[usize; 2]>,
//     angle: f32,
// ) -> (
//     Vec<[usize; 2]>,
//     Vec<Vec<[f32; INDIM as usize]>>,
//     Vec<[f32; NUM_POINTS]>,
//     Vec<[f32; 4]>,
// ) {
//     // let indices = get_random_screen_coords(NUM_RAYS);
//     let mut gold: Vec<[f32; 4]> = Vec::new();
//     for [y, x] in indices.clone() {
//         let y_ = y as f32 / HEIGHT as f32;
//         let x_ = x as f32 / WIDTH as f32;

//         let distance_from_center = ((x_ - 0.5) * (x_ - 0.5) + (y_ - 0.5) * (y_ - 0.5)).sqrt();
//         let distance_from_center_left =
//             ((x_ - 0.25) * (x_ - 0.25) + (y_ - 0.5) * (y_ - 0.5)).sqrt();
//         let distance_from_center_right =
//             ((x_ - 0.75) * (x_ - 0.75) + (y_ - 0.5) * (y_ - 0.5)).sqrt();

//         // if angle == 0. {
//         //     gold.push([1., 0., 0., 1.]);
//         // } else if f32::abs(angle - f32::consts::FRAC_PI_2) < TOL {
//         //     gold.push([0., 1., 0., 1.]);
//         // } else if (f32::abs(angle - 3. * f32::consts::FRAC_PI_2) < TOL) {
//         //     gold.push([0., 0., 1., 1.]);
//         // } else {
//         //     gold.push([1., 1., 1., 1.]);
//         // }
//         // continue;

//         if distance_from_center < 0.25
//             || (f32::abs(angle - f32::consts::FRAC_PI_2) < TOL
//                 || f32::abs(angle - 3. * f32::consts::FRAC_PI_2) < TOL)
//                 && (distance_from_center_left < 0.125 || distance_from_center_right < 0.125)
//         {
//             gold.push([1., 1., 1., 1.]);
//         } else {
//             gold.push([0., 0., 0., 0.]);
//         }
//     }

//     let (query_points, distances) =
//         sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, true);

//     return (indices, query_points, distances, gold);
// }

// gets query points for random screen coords and views for training
// pub fn get_train_batch(
//     imgs: &Vec<Vec<[f32; 4]>>,
//     iter: usize,
// ) -> (
//     Vec<[usize; 2]>,
//     Vec<Vec<[f32; INDIM as usize]>>,
//     Vec<[f32; NUM_POINTS]>,
//     Vec<[f32; 4]>,
// ) {
//     // let mut rng = thread_rng(); // Get a thread-local random number generator
//     // imgs.shuffle(&mut rng);
//     //
//     let n = iter % imgs.len(); // if we're shuffling views - angles should change accordingly
//     let angle = (n as f32 / imgs.len() as f32) * 2. * std::f32::consts::PI;

//     let indices = get_random_screen_coords(NUM_RAYS);
//     //        let screen_coords: Vec<[f32; model_tch::INDIM]> = indices
//     //            .iter()
//     //            .map(input_transforms::scale_by_screen_size_and_fourier::<3>)
//     //            .collect();

//     let (query_points, distances) =
//         sample_and_rotate_ray_points_for_screen_coords(&indices, NUM_POINTS, angle, true); // need mix rays from multiple views

//     let gold: Vec<[f32; 4]> = indices
//         .iter()
//         .map(|[y, x]| imgs[n][y * WIDTH + x])
//         .collect();

//     (indices, query_points, distances, gold)
// }
