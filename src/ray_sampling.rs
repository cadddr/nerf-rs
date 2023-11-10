use vecmath::{vec3_add, vec3_sub, vec3_normalized, vec3_dot, vec3_cross, vec3_len, vec3_scale};
use rand::random;

const HITHER: f32 = 0.05;
const FOV: f32 =  std::f32::consts::PI / 4.;

const UP: [f32; 3] =   [0., 1.,  0.];
const FROM: [f32; 3] = [0., 0., -1.];
const AT: [f32; 3] =   [0., 0.,  1.];

const NUM_SAMPLES: usize = 1;
const RAY_PROB: f32 = 200. /(512. * 512.);
const T_FAR: f32 = 10.;

pub const WIDTH: usize = 512;
pub const HEIGHT: usize = 512;

fn screen_space_to_world_space(x: f32, y: f32, width: f32, height: f32) -> [f32; 3] {
    let off: f32 = f32::tan(FOV / 2.) * HITHER;
    let offset_left = off - 2. * off * x / width;
    let offset_up = off - 2. * off * y / height;

    let view = vec3_normalized(vec3_sub(AT, FROM));
    let left = vec3_normalized(vec3_cross(view, UP));

    let to = vec3_normalized(
            vec3_add(
                    vec3_add(
                            vec3_scale(view, HITHER),
                    vec3_scale(left, offset_left)
                    ),
            vec3_scale(UP, offset_up)
            )
    );

    return to;
}

fn sample_points_along_ray(from: [f32; 3], to: [f32; 3]) -> Vec<[f32; 3]> {
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut locations: Vec<f32> = Vec::new();
    for i in 0..NUM_SAMPLES {
        let t = random::<f32>() * T_FAR;
        let point = //vec3_add(from,
        vec3_scale(to, t);
        //);
        points.push(point);
        locations.push(t);
    }

    let mut both = points.into_iter().zip(locations.into_iter()).collect::<Vec<([f32; 3], f32)>>();
    both.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    points = both.iter().map(|a| a.0).collect::<Vec<[f32; 3]>>();
    return points;
}

pub fn sample_points_along_view_directions() -> (Vec<[usize; 2]>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
    //returns xyz th, phi
    //TODO: returning view vectors rather than angles
    let mut indices: Vec<[usize; 2]> = Vec::new();
    let mut views: Vec<[f32; 3]> = Vec::new();
    let mut points: Vec<[f32; 3]> = Vec::new();

    for y in 0..HEIGHT {
        for x in 0..WIDTH {
            if random::<f32>() <= RAY_PROB {
                //TODO: rewrite as vectorized
                let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
                indices.push([y, x]);
                views.push(to);
                points.append(&mut sample_points_along_ray(FROM, to));
            }
        }
    }
    return (indices, views, points);
}

pub fn sample_points_batch_along_view_directions(batch_size: usize) -> (Vec<[usize; 2]>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
    //returns xyz th, phi
    //TODO: returning view vectors rather than angles
    let mut indices: Vec<[usize; 2]> = Vec::new();
    let mut views: Vec<[f32; 3]> = Vec::new();
    let mut points: Vec<[f32; 3]> = Vec::new();

    for i in 0..batch_size {
        let y: usize = (random::<f32>() * HEIGHT as f32) as usize;
        let x: usize = (random::<f32>() * WIDTH as f32) as usize;

        let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
        indices.push([y as usize, x as usize]);
        views.push(to);
        points.append(&mut sample_points_along_ray(FROM, to));
    }
    return (indices, views, points);
}

use tch::{Tensor, kind, Kind};
pub fn sample_points_tensor_along_view_directions(batch_size: usize) -> (Vec<[usize; 2]>, Vec<[f32; 3]>, Vec<[f32; 3]>) {
	let mut coord_y: Vec<i64> = Vec::try_from(Tensor::randint(HEIGHT as i64, &[batch_size as i64], kind::FLOAT_CPU)).unwrap(); 
	let mut coord_x: Vec<i64> = Vec::try_from(Tensor::randint(WIDTH as i64, &[batch_size as i64], kind::FLOAT_CPU)).unwrap(); 
	
	let mut indices: Vec<[usize; 2]> = coord_y.iter().zip(coord_x.iter()).map(|(y, x)|[*y as usize, *x as usize]).collect();
    let mut views: Vec<[f32; 3]> = Vec::new(); // TODO:
    let mut points: Vec<[f32; 3]> = indices.iter().map(|[y, x]| screen_space_to_world_space(*x as f32, *y as f32, WIDTH as f32, HEIGHT as f32)).map(|to| sample_points_along_ray(FROM, to)[0]).collect();
	
	return (indices, views, points);
}


#[test]
fn ray_direction_within_fov() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("x={} y={}", x, y);
    let mut to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    to[1] = 0.; //align in FOV plane
    let angle = vec3_dot(to, AT);
    println!("Cos {} vs Fov {}", angle, <f32>::cos(FOV/2.));

    assert!(angle >= <f32>::cos(FOV/2.))
}

#[test]
fn points_sampled_lie_on_ray() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("{} {}", x, y);
    let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    let points = sample_points_along_ray(FROM, to);

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|&it| {
        println!("^to {}", vec3_dot(vec3_normalized(it), to));
    });

    points.iter().for_each(|&it| {
        println!("|-to| {}", vec3_len(vec3_sub(vec3_normalized(it), to)));
    });

    assert!(points.iter().all(|&p| vec3_len(vec3_sub(vec3_normalized(p), to)) < 1e-6));
}

#[test]
fn points_sampled_ordered_by_t() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("{} {}", x, y);
    let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    let points = sample_points_along_ray(FROM, to);

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|&it| {
        println!("len {}", vec3_len(it));
    });

    let locations = points.iter().map(|&it| vec3_len(it)).collect::<Vec<f32>>();
    assert!((0..locations.len() - 1).all(|i| locations[i] <= locations[i + 1]));
}