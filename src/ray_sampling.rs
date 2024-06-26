use rand::random;
use tch::{kind, Kind, Tensor};
use vecmath::{
    col_mat4x3_transform_pos3, row_mat3x4_transform_pos3, vec3_add, vec3_cross, vec3_dot, vec3_len,
    vec3_normalized, vec3_scale, vec3_sub,
};

const HITHER: f32 = 0.05;
const FOV: f32 = std::f32::consts::PI / 4.;

const UP: [f32; 3] = [0., 1., 0.];
const FROM: [f32; 3] = [0., 0., -1.];
const AT: [f32; 3] = [0., 0., 1.];

pub const T_FAR: f32 = 10.;

pub const WIDTH: usize = 128;
pub const HEIGHT: usize = 128;

fn rotate(vec: [f32; 3], angle: f32) -> [f32; 3] {
    let c = f32::cos(angle);
    let s = f32::sin(angle);
    let rot = [
        [c, 0., s, 0.],
        [0., 1., 0., 0.],
        [-s, 0., c, 0.],
        //        [0., 0.,  0., 1.]
    ];

    row_mat3x4_transform_pos3(rot, vec)
    //    from = multvec3(rot, from.sub(at)).add(at);
}

fn screen_space_to_world_space(x: f32, y: f32, width: f32, height: f32) -> [f32; 3] {
    let off: f32 = f32::tan(FOV / 2.) * HITHER;
    let offset_left = off - 2. * off * x / width;
    let offset_up = off - 2. * off * y / height;

    let view = vec3_normalized(vec3_sub(AT, FROM));
    let left = vec3_normalized(vec3_cross(view, UP));

    let to = vec3_normalized(vec3_add(
        vec3_add(vec3_scale(view, HITHER), vec3_scale(left, offset_left)),
        vec3_scale(UP, offset_up),
    ));

    return to;
}

fn sample_points_along_ray(
    from: [f32; 3],
    to: [f32; 3],
    NUM_SAMPLES: usize,
) -> Vec<([f32; 3], f32)> {
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut locations: Vec<f32> = Vec::new();

    // 0.5 * T_FAR;
    for i in 0..NUM_SAMPLES {
        let t = random::<f32>();
        let point = //vec3_add(from,
        vec3_scale(to, t);
        //);
        points.push(point);
        locations.push(t);
    }

    let mut points_locations = points
        .into_iter()
        .zip(locations.into_iter())
        .collect::<Vec<([f32; 3], f32)>>();

    points_locations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    points_locations
    // .windows(2)
    // .map(|w| (w[1].1 - w[0].1))
    // .collect()

    // return points_locations
    //     .iter()
    //     .map(|a| [a.0[0], a.0[1], a.0[2], a.1])
    //     .collect::<Vec<[f32; 4]>>();
}

pub fn sample_points_tensor_for_rays(
    indices: Vec<[usize; 2]>,
    num_points: usize,
    angle: f32,
) -> Vec<Vec<([f32; 3], f32)>> {
    indices
        .iter()
        .map(|[y, x]| {
            screen_space_to_world_space(*x as f32, *y as f32, WIDTH as f32, HEIGHT as f32)
        })
        .map(|to| {
            sample_points_along_ray(FROM, to, num_points)
                .iter()
                .map(|pt| (rotate(pt.0, angle), pt.1))
                .collect()
        })
        .collect()
}

pub fn sample_points_tensor_along_view_directions(
    num_rays: usize,
    num_points: usize,
    angle: f32,
) -> (Vec<[usize; 2]>, Vec<[f32; 3]>, Vec<Vec<([f32; 3], f32)>>) {
    let mut coord_y: Vec<i64> =
        // Vec::try_from(10. * Tensor::arange(num_rays as i64, kind::FLOAT_CPU))
            Vec::try_from(Tensor::randint(
                HEIGHT as i64,
                &[num_rays as i64],
                kind::FLOAT_CPU,
            ))
            .unwrap();

    let mut coord_x: Vec<i64> =
        // Vec::try_from(10. * Tensor::arange(num_rays as i64, kind::FLOAT_CPU))
            Vec::try_from(Tensor::randint(
                WIDTH as i64,
                &[num_rays as i64],
                kind::FLOAT_CPU,
            ))
            .unwrap();

    let mut indices: Vec<[usize; 2]> = coord_y
        .iter()
        .zip(coord_x.iter())
        .map(|(y, x)| [*y as usize, *x as usize])
        .collect();
    let mut views: Vec<[f32; 3]> = Vec::new(); // TODO:
    let mut points: Vec<Vec<([f32; 3], f32)>> = indices
        .iter()
        .map(|[y, x]| {
            screen_space_to_world_space(*x as f32, *y as f32, WIDTH as f32, HEIGHT as f32)
        })
        .map(|to| {
            sample_points_along_ray(FROM, to, num_points)
                .iter()
                .map(|pt| (rotate(pt.0, angle), pt.1))
                .collect()
        })
        .collect();

    return (indices, views, points);
}

#[test]
fn point_rotates_to_90() {
    let angle = std::f32::consts::PI / 2.;
    let vec = [1., 2., 3.];
    println!("{:?}", rotate(vec, angle));
    assert!(rotate(vec, angle) == [3.0, 2.0, -1.0000001])
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
    println!("Cos {} vs Fov {}", angle, <f32>::cos(FOV / 2.));

    assert!(angle >= <f32>::cos(FOV / 2.))
}

// #[test]
// fn points_sampled_lie_on_ray() {
//     let x = random::<f32>() * (WIDTH as f32);
//     let y = random::<f32>() * (HEIGHT as f32);
//     println!("{} {}", x, y);
//     let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
//     println!("{:?}", to);
//     let points = sample_points_along_ray(FROM, to);

//     points.iter().for_each(|it| {
//         println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
//     });

//     points.iter().for_each(|&it| {
//         println!("^to {}", vec3_dot(vec3_normalized(it), to));
//     });

//     points.iter().for_each(|&it| {
//         println!("|-to| {}", vec3_len(vec3_sub(vec3_normalized(it), to)));
//     });

//     assert!(points
//         .iter()
//         .all(|&p| vec3_len(vec3_sub(vec3_normalized(p), to)) < 1e-6));
// }

// #[test]
// fn points_sampled_ordered_by_t() {
//     let x = random::<f32>() * (WIDTH as f32);
//     let y = random::<f32>() * (HEIGHT as f32);
//     println!("{} {}", x, y);
//     let to = screen_space_to_world_space(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
//     println!("{:?}", to);
//     let points = sample_points_along_ray(FROM, to);

//     points.iter().for_each(|it| {
//         println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
//     });

//     points.iter().for_each(|&it| {
//         println!("len {}", vec3_len(it));
//     });

//     let locations = points.iter().map(|&it| vec3_len(it)).collect::<Vec<f32>>();
//     assert!((0..locations.len() - 1).all(|i| locations[i] <= locations[i + 1]));
// }
