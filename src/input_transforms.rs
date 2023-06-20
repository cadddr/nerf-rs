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
		e[0] - 0.5,
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
	    (1. - e[1]),
	    f32::sqrt(-ec[0] * (-ec[0] + ec[1] * ec[1])),
	    1. / (f32::tan(-ec[0] / (ec[1] + 1e-6) + 1e-6))
	]
}

