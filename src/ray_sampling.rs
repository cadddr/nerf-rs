use crate::model::NUM_POINTS;
use rand::random;
use vecmath::{
    row_mat3x4_transform_pos3, vec2_cross, vec3_add, vec3_cross, vec3_dot, vec3_len,
    vec3_normalized, vec3_scale, vec3_sub,
};
const HITHER: f32 = 0.05;
const FOV: f32 = std::f32::consts::PI / 3.;

const UP: [f32; 3] = [0., 1., 0.];
const FROM: [f32; 3] = [0., 0., -1.];
const AT: [f32; 3] = [0., 0., 1.];

pub const T_FAR: f32 = 2.;

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

fn screen_to_world(x: f32, y: f32, width: f32, height: f32) -> [f32; 3] {
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

fn sample_points_along_ray_and_rotate(
    from: [f32; 3],
    to: [f32; 3],
    view_angle: f32,
    num_samples: usize,
) -> (Vec<[f32; 3]>, [f32; NUM_POINTS]) {
    let mut points: Vec<[f32; 3]> = Vec::new();
    let mut locations: Vec<f32> = Vec::new();

    for _ in 0..num_samples {
        let t = random::<f32>() * (T_FAR - HITHER) + HITHER;
        let point = vec3_add(from, vec3_scale(to, t)); // add back origin of view vector to get point's world coordinates
        points.push(point);
        locations.push(t);
    }

    let mut points_locations = points
        .into_iter()
        .zip(locations.into_iter())
        .collect::<Vec<([f32; 3], f32)>>();

    points_locations.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // unzip back after sorting by t
    points = points_locations
        .iter()
        .map(|([x, y, z], _)| rotate([*x, *y, *z], view_angle)) // and rotate for view angle
        .collect::<Vec<[f32; 3]>>();

    locations = points_locations
        .iter()
        .map(|(_, t)| *t)
        .collect::<Vec<f32>>()
        .try_into()
        .unwrap();

    return (points, locations.try_into().unwrap());
}

pub fn sample_and_rotate_rays_for_screen_coords(
    indices: &Vec<[usize; 2]>,
    angle: f32,
) -> Vec<[f32; 3]> {
    let rays: Vec<[f32; 3]> = indices
        .iter()
        .map(|[y, x]| screen_to_world(*x as f32, *y as f32, WIDTH as f32, HEIGHT as f32))
        .map(|vec| rotate(vec, angle))
        .collect();
    return rays;
}

pub fn sample_and_rotate_ray_points_for_screen_coords(
    indices: &Vec<[usize; 2]>,
    num_points: usize,
    angle: f32,
) -> (Vec<Vec<[f32; 3]>>, Vec<[f32; NUM_POINTS]>) {
    let rays: Vec<[f32; 3]> = indices
        .iter()
        .map(|[y, x]| screen_to_world(*x as f32, *y as f32, WIDTH as f32, HEIGHT as f32))
        .collect();

    let (points, locations) = rays
        .iter()
        .map(|to| sample_points_along_ray_and_rotate(FROM, *to, angle, num_points))
        .collect();

    return (points, locations);
}

// '''cam, view, segment start, segment end'''
//     '''
//     c+(v-c)*p = a+(b-a)*t
//     '''
pub fn ray_intersection(
    c: [f32; 3],
    v: [f32; 3],
    a: [f32; 3],
    b: [f32; 3],
) -> (f32, f32, [f32; 3], [f32; 3]) {
    let l = vec3_sub(v, c);
    // println!("v-c {:?}", l);
    let m = vec3_sub(b, a);
    // println!("b-a {:?}", m);
    // try:
    //     t = ((b-a).cross(c-a)).mag() / (b-a).dot(PVector(-l.y,l.x,0))
    //     p = (c-a).dot(PVector(-l.y,l.x,0))/(b-a).dot(PVector(-l.y,l.x,0))
    //     // # if p > 0 and p < 9000 and t > 0 and t < 1:
    //     return t, p
    // except:
    //     return -1,-1
    let l_ = vec3_cross(l, m); //[-l.get(1).unwrap(), *l.get(0).unwrap(), 0.];
                               // println!("l x m {:?}", l_);
                               // println!("m . l_ {:?}", vec3_dot(m, l_));
    let t = vec3_len(vec3_cross(m, vec3_sub(c, a))) / vec3_len(l_); //vec3_dot(m, l_);
    let p = vec3_len(vec3_cross(vec3_sub(c, a), l)) / vec3_len(l_); //vec3_dot(m, l_);

    // if p > 0. && p < 9000. && t > 0. && t < 1. {
    return (
        t,
        p,
        vec3_add(c, vec3_scale(l, t)),
        vec3_add(a, vec3_scale(m, p)),
    );
    // }
    // return (-1., -1.);
}

#[test]
fn test_intersection() {
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, [-1., 0., -1.], [1., 0., 1.]) // diagonal -x-z to xz
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, [1., 0., 1.], [-1., 0., -1.]) // diagonal xz to -x-z
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, UP, vec3_scale(UP, -2.)) // centered vertical y to -2y
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, [-1., 0., -1.], [1., 0., -1.]) // across -x to x at -z
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, [-1., 0., 1.], [1., 0., 1.]) // across -x to x at z
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, [-1., 0., 1.], [-1., 0., -1.]) // counterexample, parallel to view offset to -x
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(FROM, AT, [-1., -1., 0.], [1., 1., 0.]) // diagonal -x-y to xy
    );
    println!(
        "a+(b-a)*t, c+(v-c)*p {:?}",
        ray_intersection(
            FROM,
            AT,
            vec3_add([-1., -1., 0.], [0.5, 0., 0.]),
            vec3_add([1., 1., 0.], [0.5, 0., 0.])
        ) // counterexample diagonal -x-y to xy, offset by small value
    );
}

#[test]
fn test_cross_determinant() {
    println!("{:?}", vec2_cross([7., 5.], [3., 2.])); // 7*2 - 3*5 = -1
    println!("{:?}", vec3_cross([7., 5., 0.], [3., 2., 0.])); // 7*2 - 3*5 = -1
}

#[test]
fn ray_direction_within_fov() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("x={} y={}", x, y);
    let mut to = screen_to_world(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("{:?}", to);
    to[1] = 0.; //align in FOV plane
    let angle = vec3_dot(to, AT);
    println!("Cos {} vs Fov {}", angle, <f32>::cos(FOV / 2.));

    assert!(angle >= <f32>::cos(FOV / 2.))
}

#[test]
fn points_sampled_lie_on_ray() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("coords {} {}", x, y);
    let to = screen_to_world(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("to {:?}", to);
    let (points, _) = sample_points_along_ray_and_rotate(FROM, to, 0., NUM_POINTS);

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    points.iter().for_each(|&it| {
        println!(
            "dot(p, to) {}",
            vec3_dot(vec3_normalized(vec3_sub(it, FROM)), to)
        );
    });

    points.iter().for_each(|&it| {
        println!(
            "|sub(p, to)| {}",
            vec3_len(vec3_sub(vec3_normalized(vec3_sub(it, FROM)), to))
        );
    });

    assert!(points
        .iter()
        .all(|&p| vec3_len(vec3_sub(vec3_normalized(vec3_sub(p, FROM)), to)) < 1e-6));
}

#[test]
fn points_sampled_ordered_by_t() {
    let x = random::<f32>() * (WIDTH as f32);
    let y = random::<f32>() * (HEIGHT as f32);
    println!("coords {} {}", x, y);
    let to = screen_to_world(x as f32, y as f32, WIDTH as f32, HEIGHT as f32);
    println!("to {:?}", to);
    let (points, locations) = sample_points_along_ray_and_rotate(FROM, to, 0., NUM_POINTS);

    points.iter().for_each(|it| {
        println!("point {{ {:.2}, {:.2}, {:.2} }}", it[0], it[1], it[2]);
    });

    locations.iter().for_each(|t| {
        println!("t {:.2}", t);
    });

    points.iter().for_each(|&it| {
        println!("len {}", vec3_len(vec3_sub(it, FROM)));
    });

    let lengths = points
        .iter()
        .map(|&it| vec3_len(vec3_sub(it, FROM)))
        .collect::<Vec<f32>>();
    assert!((0..lengths.len() - 1).all(|i| lengths[i] <= lengths[i + 1]));
    assert!((0..locations.len() - 1).all(|i| locations[i] <= locations[i + 1]));
}

#[test]
fn point_rotates_to_90() {
    let angle = std::f32::consts::PI / 2.;
    let vec = [1., 2., 3.];
    println!("{:?}", rotate(vec, angle));
    assert!(rotate(vec, angle) == [3.0, 2.0, -1.0000001])
}
