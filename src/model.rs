use crate::ray_sampling::T_FAR;
use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use tch::nn::ModuleT;
use tch::{nn, nn::Optimizer, nn::OptimizerConfig, Device, Kind, Tensor};

pub const NUM_RAYS: usize = 16384;
pub const NUM_POINTS: usize = 16;
pub const BATCH_SIZE: i64 = NUM_RAYS as i64 * NUM_POINTS as i64;

pub const INDIM: i64 = 3;
const HIDDEN_NODES: i64 = 100;
pub const LABELS: i64 = 4;

pub fn hparams() -> HashMap<String, f32> {
    let mut map: HashMap<String, f32> = HashMap::new();
    map.insert("NUM_RAYS".to_string(), NUM_RAYS as f32);
    map.insert("NUM_POINTS".to_string(), NUM_POINTS as f32);
    map.insert("BATCH_SIZE".to_string(), BATCH_SIZE as f32);
    map.insert("INDIM".to_string(), INDIM as f32);
    map.insert("HIDDEN_NODES".to_string(), HIDDEN_NODES as f32);
    map.insert("LABELS".to_string(), LABELS as f32);
    return map;
}

#[derive(Debug)]
pub struct DensityNet {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    fc4: nn::Linear,
    fc5: nn::Linear,
    fc6: nn::Linear,
    fc7: nn::Linear,
    fc8: nn::Linear,
}

#[derive(Debug)]
struct RadianceNet {
    fc9: nn::Linear,
    fc10: nn::Linear,
}

impl DensityNet {
    pub fn new(vs: &nn::Path) -> DensityNet {
        // the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer)
        // and outputs σ and a 256-dimensional feature vector.
        let fc1 = nn::linear(vs, INDIM, HIDDEN_NODES, Default::default());
        let fc2 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc3 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc4 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc5 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc6 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc7 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc8 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES + 1, Default::default());

        DensityNet {
            fc1,
            fc2,
            fc3,
            fc4,
            fc5,
            fc6,
            fc7,
            fc8,
        }
    }
    pub fn predict<const BATCH_SIZE: i64, const NUM_RAYS: usize, const NUM_POINTS: usize>(
        &self,
        mut coords: Tensor,
    ) -> Tensor {
        assert_eq!(coords.size(), vec![BATCH_SIZE * INDIM]);

        coords = coords.to(Device::Mps).view((BATCH_SIZE, INDIM));
        let densities_features = self.forward_t(&coords, true);

        let densities = densities_features
            .view((NUM_RAYS as i64, NUM_POINTS as i64, HIDDEN_NODES + 1))
            .permute(&[2, 0, 1])
            .get(0);
        return densities.unsqueeze(2);
    }
}

impl RadianceNet {
    fn new(vs: &nn::Path) -> RadianceNet {
        // This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional
        // fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.
        let fc9 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES / 2, Default::default());
        let fc10 = nn::linear(vs, HIDDEN_NODES / 2, LABELS, Default::default());

        RadianceNet { fc9, fc10 }
    }
}

impl nn::ModuleT for DensityNet {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let densities_features = xs
            .apply(&self.fc1)
            .relu()
            .apply(&self.fc2)
            .relu()
            .apply(&self.fc3)
            .relu()
            .apply(&self.fc4)
            .relu()
            .apply(&self.fc5)
            .relu()
            .apply(&self.fc6)
            .relu()
            .apply(&self.fc7)
            .relu()
            .apply(&self.fc8);

        return densities_features;
    }
}

impl nn::ModuleT for RadianceNet {
    fn forward_t(&self, features: &Tensor, train: bool) -> Tensor {
        let colors = features
            .view((BATCH_SIZE, HIDDEN_NODES))
            .apply(&self.fc9)
            .relu()
            .apply(&self.fc10)
            .sigmoid()
            .view((NUM_RAYS as i64, NUM_POINTS as i64, LABELS));

        colors
    }
}

pub struct NeRF {
    pub vs: nn::VarStore,
    pub density: DensityNet,
    radiance: RadianceNet,
}

impl NeRF {
    pub fn new() -> NeRF {
        let vs = nn::VarStore::new(Device::Mps);
        let density = DensityNet::new(&vs.root());
        let radiance = RadianceNet::new(&vs.root());

        NeRF {
            vs,
            density,
            radiance,
        }
    }

    pub fn predict(
        &self,
        mut query_points: Tensor, // viewing direction should only go through fc9
        mut distances: Tensor,
    ) -> (Tensor, Tensor) {
        // const INDIM_BATCHED: usize = INDIM * BATCH_SIZE;

        // let coords_flat =
        //     array_vec_to_1d_array::<INDIM, INDIM_BATCHED>(&array_vec_vec_to_array_vec(coords));
        // let coords_tensor = Tensor::of_slice(&coords_flat).view((BATCH_SIZE as i64, INDIM));
        assert_eq!(query_points.size(), vec![BATCH_SIZE * INDIM]);
        assert_eq!(distances.size(), vec![BATCH_SIZE]);

        query_points = query_points.to(Device::Mps).view((BATCH_SIZE, INDIM));
        let densities_features = self.density.forward_t(&query_points, true);

        let densities = densities_features
            .view((NUM_RAYS as i64, NUM_POINTS as i64, HIDDEN_NODES + 1))
            .permute(&[2, 0, 1])
            .get(0);

        let features = densities_features
            .view((NUM_RAYS as i64, NUM_POINTS as i64, HIDDEN_NODES + 1))
            .slice(2, 1, HIDDEN_NODES + 1, 1) // should concat with view dir, not drop density
            .view((BATCH_SIZE, HIDDEN_NODES));

        let colors = self.radiance.forward_t(&features, true);

        // let distances_flat = array_vec_to_1d_array::<NUM_POINTS, BATCH_SIZE>(&distances); // TODO: check
        // let mut distances_tensor =
        //     Tensor::of_slice(&distances_flat).view((NUM_RAYS as i64, NUM_POINTS as i64));

        let tfar = Tensor::of_slice(&[T_FAR; NUM_RAYS]).unsqueeze(1);
        distances = distances.view((NUM_RAYS as i64, NUM_POINTS as i64));
        distances =
            Tensor::concat(&[distances.slice(1, 1, NUM_POINTS as i64, 1), tfar], 1) - distances; // distances between adjacent samples, eq. (3)

        return (
            compositing(
                &densities,
                Tensor::stack(
                    &[
                        &densities,
                        &densities,
                        &densities,
                        &Tensor::ones(
                            &[NUM_RAYS as i64, NUM_POINTS as i64],
                            (Kind::Float, densities.device()),
                        ),
                    ],
                    0,
                )
                .permute(&[1, 2, 0]),
                distances.to(Device::Mps),
            ),
            densities,
        );
    }

    pub fn save(&self, save_path: &str) {
        self.vs.save(&save_path).unwrap();
    }

    pub fn load(&mut self, load_path: &str) {
        self.vs.load(&load_path).unwrap();
    }
}

// eq. (3)
fn accumulated_transmittance(densities: &Tensor, distances: &Tensor, i: i64) -> Tensor {
    if i == 0 {
        // First sample has full transmittance
        return Tensor::ones(&[NUM_RAYS as i64], (Kind::Float, densities.device()));
    }
    let result = (densities.slice(1, 0, i, 1) * distances.slice(1, 0, i, 1)) // should just be sum of densities up to i - 1
        .sum_dim_intlist(Some([1i64].as_slice()), false, Kind::Float)
        .neg()
        .exp();

    return result;
}

fn compositing(densities: &Tensor, colors: Tensor, distances: Tensor) -> Tensor {
    let tensor_vector: Vec<Tensor> = (0..NUM_POINTS)
        .map(|i| accumulated_transmittance(&densities, &distances, i as i64))
        .collect();

    let tensor_array: [Tensor; NUM_POINTS] = tensor_vector.try_into().unwrap();

    let T = Tensor::stack(&tensor_array, 0).view((NUM_RAYS as i64, NUM_POINTS as i64)); //TODO: check shaping/ordering

    let weights = (T * (1. as f32 - (densities * distances).neg().exp())).unsqueeze(2);

    let final_colors: Tensor =
        (weights * colors).sum_dim_intlist(Some([1i64].as_slice()), false, Kind::Float);

    return final_colors;
}

#[test]
fn transpose_order_test() {
    let rays = 3;
    let pts = 2;
    let lbl = 4;
    let a = Tensor::of_slice(&[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ]);
    a.reshape(&[rays, pts, lbl]).print();

    // a.reshape(&[pts, rays, lbl])
    //     // .get(0)
    //     .print(); // wrong

    a.view((rays, pts, lbl)).permute(&[1, 0, 2]).print();
}

#[test]
fn flatten_order_test() {
    let rays = 3;
    let pts = 2;
    let lbl = 4;
    let a = Tensor::of_slice(&[
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
    ]);

    a.reshape(&[rays, pts, lbl]).print();
    a.reshape(&[rays * pts, lbl]).print();
    a.reshape(&[rays * pts, lbl])
        .reshape(&[rays, pts, lbl])
        .print();
}

#[test]
fn off_by_one_slice() {
    for i in (0..NUM_POINTS) {
        println!["{:?}", i];
    }

    assert_eq![
        Tensor::of_slice(&[1, 2, 3]).slice(0, 0, 2, 1),
        Tensor::of_slice(&[1, 2, 3])
    ]
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}

pub struct Trainer {
    opt: Optimizer,
}

impl Trainer {
    pub fn new(vs: &nn::VarStore, lr: f64) -> Trainer {
        let opt = nn::Adam::default().build(&vs, lr).unwrap();
        Trainer { opt }
    }

    pub fn step(&mut self, predictions: &Tensor, mut gold: Tensor, iter: &usize) -> f32 {
        // const LABELS_BATCHED: usize = LABELS * NUM_RAYS;
        // let gold_flat = array_vec_to_1d_array::<LABELS, LABELS_BATCHED>(&gold);
        // let gold_tensor = Tensor::of_slice(&gold_flat).view((NUM_RAYS as i64, LABELS as i64));
        assert_eq!(predictions.size(), vec![NUM_RAYS as i64, LABELS]);
        assert_eq!(gold.size(), vec![NUM_RAYS as i64 * LABELS]);

        gold = gold.to(Device::Mps).view((NUM_RAYS as i64, LABELS));
        let loss = mse_loss(&predictions, &gold);

        // self.backward_scale_grad_step(&loss);
        self.opt.backward_step(&loss);
        // self.grad_accumulation_step(&loss, iter, accumulation_steps);
        return f32::try_from(&loss).unwrap();
    }

    fn grad_accumulation_step(&mut self, loss: &Tensor, iter: &usize, accumulation_steps: usize) {
        if iter % accumulation_steps == 0 {
            self.opt.zero_grad();
        }
        loss.backward();

        if iter % accumulation_steps == 0 {
            self.opt.step();
        }
    }

    fn backward_scale_grad_step(&mut self, loss: &Tensor) {
        self.opt.zero_grad();
        loss.backward();
        for var in self.opt.trainable_variables() {
            let mut grad = var.grad();
            grad *= NUM_POINTS as f32;
        }
        self.opt.step();
    }
}

// fn array_vec_vec_to_array_vec(vv: &Vec<Vec<[f32; INDIM]>>) -> Vec<[f32; INDIM]> {
//     let mut v = Vec::new();
//     for subvec in vv {
//         for el in subvec {
//             v.push(*el);
//         }
//     }
//     return v;
// }

// fn array_vec_to_1d_array<const INNER_DIM: usize, const OUT_DIM: usize>(
//     v: &Vec<[f32; INNER_DIM]>,
// ) -> [f32; OUT_DIM] {
//     let mut array = [0f32; OUT_DIM];

//     for batch_index in 0..OUT_DIM / INNER_DIM {
//         for item_index in 0..INNER_DIM {
//             array[batch_index * INNER_DIM + item_index] = v[batch_index][item_index];
//         }
//     }
//     return array;
// }

pub fn tensor_from_3d<const INNER_DIM: usize>(query_points: &Vec<Vec<[f32; INNER_DIM]>>) -> Tensor {
    Tensor::of_slice(
        &query_points
            .clone()
            .into_iter()
            .flatten()
            .flatten()
            .collect::<Vec<f32>>(),
    )
}

pub fn tensor_from_2d<const INNER_DIM: usize>(distances: &Vec<[f32; INNER_DIM]>) -> Tensor {
    Tensor::of_slice(
        &distances
            .clone()
            .into_iter()
            .flatten()
            .collect::<Vec<f32>>(),
    )
}

pub fn tensor_to_2d(a: &Tensor) -> Vec<Vec<f32>> {
    return Vec::<Vec<f32>>::try_from(a.to_kind(Kind::Float).to_device(Device::Cpu)).unwrap();
}

#[test]
pub fn test_flatten_to_tensor() {
    let x = vec![
        vec![&[1i64, 2i64]],
        vec![&[1i64, 2i64]],
        vec![&[1i64, 2i64]],
    ];
    let y = x.into_iter().flatten().collect::<Vec<&[i64; 2]>>();
    let z = y.into_iter().flatten().collect::<Vec<&i64>>();
    println!["{:?}", z.len()];
}
