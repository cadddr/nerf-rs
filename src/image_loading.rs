use image;
use load_image::ImageData;

use std::{env, fs};

pub fn load_image_as_array(path: &str) -> Vec<[f32; 4]> {
    let img = load_image::load_image(path, false).unwrap();
    let mut pixels: Vec<[f32; 4]> = Vec::new();
    if let ImageData::RGBA8(bitmap) = img.bitmap {
        pixels = bitmap
            .iter()
            .map(|rgba| {
                [
                    rgba.r as f32 / 255.,
                    rgba.g as f32 / 255.,
                    rgba.b as f32 / 255.,
                    rgba.a as f32 / 255.,
                ]
            })
            .collect::<Vec<[f32; 4]>>();
    }
    println!("image {:?} pixels", pixels.len());
    return pixels;
}

pub fn get_image_paths_from_dir(dir: String) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();

    for entry in fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        paths.push(String::from(entry.path().to_string_lossy()));
    }

    return paths;
}

pub fn get_image_paths(dir: String, start: usize, end: usize, step: usize) -> Vec<String> {
    let mut paths: Vec<String> = Vec::new();
    assert!(start < end);
    assert!((end - start) % step == 0);
    assert!((end - start) / step > 0);

    for i in (start..end).step_by(step) {
        paths.push(format!("{}/image-{}.png", dir, i));
    }

    return paths;
}

pub fn load_multiple_images_as_arrays(paths: Vec<String>) -> Vec<Vec<[f32; 4]>> {
    //    let images = Vec::new();
    paths
        .iter()
        .map(|path| load_image_as_array(path.as_str()))
        .collect()
}

//pub fn to_u8_rgba(rgba: u32) -> [u8; 4] {
//    let (r, g, b) = (r as u32, g as u32, b as u32);
//    (r << 16) | (g << 8) | b
//}
