use load_image::ImageData;
use image;

pub fn load_image_as_array(path: &str) -> Vec<[f32; 4]> {
    let img = load_image::load_image(path, false).unwrap();
    let mut pixels: Vec<[f32; 4]> = Vec::new();
    if let ImageData::RGBA8(bitmap) = img.bitmap {
        pixels = bitmap.iter().map(|rgba| [rgba.r as f32 / 255., rgba.g as f32 / 255., rgba.b as f32 / 255., rgba.a as f32 / 255.]).collect::<Vec<[f32; 4]>>();
    }
    println!("image {:?} pixels", pixels.len());
    return pixels;
}

pub fn load_multiple_images_as_arrays(paths: Vec<&str>) -> Vec<Vec<[f32; 4]>> {
//    let images = Vec::new();
    paths.iter().map(|path| load_image_as_array(*path)).collect()
}


//pub fn to_u8_rgba(rgba: u32) -> [u8; 4] {
//    let (r, g, b) = (r as u32, g as u32, b as u32);
//    (r << 16) | (g << 8) | b
//}