use crate::display::{prediction_array_as_u32, rgba_to_u8_array};
use crate::model::NUM_POINTS;
use std::collections::HashMap;
use tensorboard_rs::summary_writer::SummaryWriter;

pub fn log_params(writer: &mut SummaryWriter, params: &HashMap<String, f32>) {
    for (key, value) in params {
        writer.add_scalar(key, *value, 0);
    }
}

pub fn log_query_points(
    writer: &mut SummaryWriter,
    query_points: &Vec<Vec<[f32; 3]>>,
    distances: &Vec<[f32; NUM_POINTS]>,
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

pub fn log_densities(
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

pub fn log_density_maps(
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

pub fn log_as_hist<const RANGE: usize>(
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
