extern crate minifb;
use minifb::{Key, Window, WindowOptions};

use crate::ray_sampling::{HEIGHT, WIDTH};

pub fn run_window<F: FnMut(&mut Vec<u32>)>(
    mut update_window_buffer: F,
    width: usize,
    height: usize,
) {
    let mut window =
        Window::new("NeRF Live Preview", width, height, WindowOptions::default()).unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_micros(1000)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut buffer: Vec<u32> = vec![0; width * height];
        update_window_buffer(&mut buffer);
        window.update_with_buffer(&buffer, width, height).unwrap();
    }
}

pub fn draw_to_screen(
    buffer: &mut Vec<u32>,
    backbuffer: &[u32; WIDTH * HEIGHT], //img: &Vec<[f32; 4]>
    debug: bool,
    gold: Vec<[f32; 4]>,
    iter: &usize,
) {
    // draw from either backbuffer or gold image
    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if debug {
                let gold_ = gold[y * WIDTH + x];
                buffer[y * WIDTH + x] = from_u8_rgb(
                    (gold_[0] * 255.) as u8,
                    (gold_[1] * 255.) as u8,
                    (gold_[2] * 255.) as u8,
                );
            } else {
                buffer[y * WIDTH + x] = backbuffer[y * WIDTH + x];
            }
        }
    }
}

pub fn from_u8_rgb(r: u8, g: u8, b: u8) -> u32 {
    let (r, g, b) = (r as u32, g as u32, b as u32);
    (r << 16) | (g << 8) | b
}

pub fn rgba_to_u8_array(rgba: &u32) -> [u8; 3] {
    [(*rgba >> 16) as u8, (*rgba >> 8) as u8, *rgba as u8]
}

pub fn prediction_array_as_u32(rgba: &[f32; 4]) -> u32 {
    return from_u8_rgb(
        (rgba[0] * 255.) as u8,
        (rgba[1] * 255.) as u8,
        (rgba[2] * 255.) as u8,
    );
}
