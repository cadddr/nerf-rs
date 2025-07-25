extern crate minifb;
use minifb::{Key, Window, WindowOptions};

pub fn run_window<F: FnMut(&mut Vec<u32>)>(mut update_window_buffer: F, WIDTH: usize, HEIGHT: usize) {
    let mut window = Window::new("Test - ESC to exit", WIDTH, HEIGHT, WindowOptions::default()).unwrap();
    window.limit_update_rate(Some(std::time::Duration::from_micros(1000)));

    while window.is_open() && !window.is_key_down(Key::Escape) {
        let mut buffer: Vec<u32> = vec![0; WIDTH * HEIGHT];
        update_window_buffer(&mut buffer);
        window.update_with_buffer(&buffer, WIDTH, HEIGHT).unwrap();
    }
}
