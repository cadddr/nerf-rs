use crate::ray_sampling::{HEIGHT, WIDTH};

pub fn identity(e: &[usize; 2]) -> [f32; 2] {
	[
		e[0] as f32, 
		e[1] as f32
	]
}

pub fn scale_by_screen_size_and_center(e: &[usize; 2]) -> [f32; 2] {
	center(&scale_by_screen_size(e))
}

pub fn scale_by_screen_size(e: &[usize; 2]) -> [f32; 2] {
	[
		e[0] as f32 / HEIGHT as f32, 
		e[1] as f32 / WIDTH as f32
	]
}

fn center(e: &[f32; 2]) -> [f32; 2] {
	[
		1. - e[0] - 0.5,
	    e[1] - 0.5
	]
}

pub fn scale_by_screen_size_and_coconet(e: &[usize; 2]) -> [f32; 6] {
	corners_and_polar(&scale_by_screen_size(&e))
}

fn corners_and_polar(e: &[f32; 2]) -> [f32; 6] {
	let ec = center(&e);
	[
		e[0],
	    e[1],
	    (1. - e[0]),
		(1.- e[1]),
	    f32::sqrt(ec[0] * ec[0] + ec[1] * ec[1]),
	    1. / (f32::tan(ec[0] / (ec[1] + 1e-6) + 1e-6))
	]
}

pub fn scale_by_screen_size_and_fourier<const L: usize>(e: &[usize; 2]) -> [f32; L] {
	fourier_features::<L>(&scale_by_screen_size_and_center(&e))
}

fn fourier_features<const L: usize>(e: &[f32; 2]) -> [f32; L] {
	let mut encoding = [0f32; L];
	for i in 0..L / 2 {
		if i % 2 == 0 {
			encoding[i] = f32::sin(f32::powf(2., (i / 2) as f32) * e[1]);
		}
		else {
			encoding[i] = f32::cos(f32::powf(2., (i / 2) as f32) * e[0]);
		}
	}
	return encoding;
}

