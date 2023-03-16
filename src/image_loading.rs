use load_image::ImageData;
use image;

pub fn load_image_as_array(path: &str) -> Vec<[f32; 4]> {
    let img = load_image::load_image(path, false).unwrap();
    let mut pixels: Vec<[f32; 4]> = Vec::new();
    if let ImageData::RGBA8(bitmap) = img.bitmap {
        pixels = bitmap.iter().map(|rgba| [rgba.r as f32 / 255., rgba.g as f32 / 255., rgba.b as f32 / 255., rgba.a as f32 / 255.]).collect::<Vec<[f32; 4]>>();
    }
    return pixels;
}

//pub fn save_array_as_image(pixels: &[u32], path: &str) {
//    let buffer = to_u8_rgba(rgba)
//    image::save_buffer(&Path::new(path), buffer, 128, 128, image::RGBA(8))
//}
//
//pub fn to_u8_rgba(rgba: u32) -> [u8; 4] {
//    let (r, g, b) = (r as u32, g as u32, b as u32);
//    (r << 16) | (g << 8) | b
//}