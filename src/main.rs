//We synthesize images by sampling 5D coordinates (location and viewing direction) along camera rays (a),
//feeding those locations into an MLP to produce a color and volume density (b),
//and using volume ren- dering techniques to composite these values into an image (c)

use std;
use rand::random;
use vecmath::{vec3_add, vec3_sub, vec3_normalized, vec3_dot, vec3_cross, vec3_len, vec3_scale};

const HITHER: f32 = 0.05;
const FOV: f32 = 3.14 / 4.;

const UP: [f32; 3] =   [0., 1.,  0.];
const FROM: [f32; 3] = [0., 0., -1.];
const AT: [f32; 3] =   [0., 0.,  1.];

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

const NUM_SAMPLES: usize = 3;
const RAY_PROB: f32 = 10./(512. * 512.);
const T_FAR: f32 = 10.;

fn sample_points_along_ray(from: [f32; 3], to: [f32; 3]) -> Vec<[f32; 3]> {
    let mut points: Vec<[f32; 3]> = Vec::new();
    for i in 0..NUM_SAMPLES {
        let t = random::<f32>() * T_FAR;
        let point = //vec3_add(from,
        vec3_scale(to, t);
        //);
        points.push(point)
    }
    return points;
}

const WIDTH: usize = 512;
const HEIGHT: usize = 512;

fn sample_points_along_view_directions() -> (Vec<[f32; 3]>, Vec<[f32; 3]>) {
    //returns xyz th, phi
    let mut views: Vec<[f32; 3]> = Vec::new();
    let mut points: Vec<[f32; 3]> = Vec::new();

    for y in (0..HEIGHT) {
        for x in (0..WIDTH) {
            if random::<f32>() <= RAY_PROB {
                //TODO: rewrite as vectorized
                let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
                views.push(to);
                points.append(&mut sample_points_along_ray(FROM, to));
            }
        }
    }
    return (views, points);
}

fn predict_emittance_and_density() {

}

fn accumulate_radiance() {

}

fn main() {
    let (views, points) = sample_points_along_view_directions();

    views.iter().for_each(|it| {
        println!("vector {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });
}

#[test]
fn ray_direction_within_fov() {
    //TODO: align in FOV plane
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("{} {}", x, y);
    let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    let angle = vec3_dot(to, AT);
    println!("{} vs {}", angle, <f32>::cos(FOV/2.));

    assert!(angle >= <f32>::cos(FOV/2.))
}

#[test]
fn points_sample_lie_on_ray() {
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
