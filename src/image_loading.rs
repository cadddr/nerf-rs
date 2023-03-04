use load_image::ImageData;

pub fn load_image_as_array(path: &str) -> Vec<[f32; 4]> {
    let img = load_image::load_image(path, false).unwrap();
    let mut pixels: Vec<[f32; 4]> = Vec::new();
    if let ImageData::RGBA8(bitmap) = img.bitmap {
        pixels = bitmap.iter().map(|rgba| [rgba.r as f32 / 255., rgba.g as f32 / 255., rgba.b as f32 / 255., rgba.a as f32 / 255.]).collect::<Vec<[f32; 4]>>();
    }
    return pixels;
}