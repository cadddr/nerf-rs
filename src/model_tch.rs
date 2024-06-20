use dfdx::tensor_ops::sigmoid;
use tch::{
    nn, nn::Module, nn::Optimizer, nn::OptimizerConfig, nn::Sequential, Device, Kind, Tensor,
};

pub const NUM_RAYS: usize = 16384;
pub const NUM_POINTS: usize = 2;
pub const BATCH_SIZE: usize = NUM_RAYS * NUM_POINTS;

pub const INDIM: usize = 4;
const HIDDEN_NODES: i64 = 100;
const LABELS: usize = 4;

#[derive(Debug)]
struct Net {
    fc1: nn::Linear,
    fc2: nn::Linear,
    fc3: nn::Linear,
    fc4: nn::Linear,
    fc5: nn::Linear,
    fc6: nn::Linear,
    fc7: nn::Linear,
    fc8: nn::Linear,
    fc9: nn::Linear,
}

impl Net {
    fn new(vs: &nn::Path) -> Net {
        // the MLP FΘ first processes the input 3D coordinate x with 8 fully-connected layers (using ReLU activations and 256 channels per layer)
        // and outputs σ and a 256-dimensional feature vector.
        let fc1 = nn::linear(vs, INDIM as i64, HIDDEN_NODES, Default::default());
        let fc2 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc3 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc4 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc5 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc6 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc7 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES, Default::default());
        let fc8 = nn::linear(vs, HIDDEN_NODES, HIDDEN_NODES + 1, Default::default());
        // This feature vector is then concatenated with the camera ray’s viewing direction and passed to one additional
        // fully-connected layer (using a ReLU activation and 128 channels) that output the view-dependent RGB color.
        let fc9 = nn::linear(vs, HIDDEN_NODES, LABELS as i64, Default::default());
        Net {
            fc1,
            fc2,
            fc3,
            fc4,
            fc5,
            fc6,
            fc7,
            fc8,
            fc9,
        }
    }
}

impl nn::ModuleT for Net {
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
            .apply(&self.fc8)
            .relu();

        return densities_features;

        // Tensor::narrow

        // .sigmoid()
        // .view((NUM_RAYS as i64, LABELS as i64, NUM_POINTS as i64))
        // // .mean_dim(Some([1i64].as_slice()), false, Kind::Float)
        // .avg_pool1d(
        //     &[NUM_POINTS as i64],
        //     &[1 as i64],
        //     &[0 as i64],
        //     // &[1 as i64],
        //     false,
        //     false,
        // )
        // .view((NUM_RAYS as i64, LABELS as i64));
    }
}

fn accumulated_transmittance(densities: &Tensor, distances: &Tensor, i: i64) -> Tensor {
    let result = (densities.slice(1 as i64, 0, i - 1, 1) * distances.slice(1 as i64, 0, i - 1, 1))
        .sum_dim_intlist(Some([1i64].as_slice()), false, Kind::Float)
        .exp();

    return result;
}

fn mean_compositing(colors: Tensor) -> Tensor {
    colors
        // .view((NUM_RAYS as i64, NUM_POINTS as i64, LABELS as i64))
        .mean_dim(Some([1i64].as_slice()), false, Kind::Float)
}

fn select_compositing(colors: Tensor) -> Tensor {
    println!("before {:?}", colors);
    colors.get(0).print();

    let result = colors
        // .view((NUM_POINTS as i64, NUM_RAYS as i64, LABELS as i64))
        .permute(&[1, 0, 2])
        .get(1);

    println!("after {:?}", result);
    result.get(0).print();

    println!(
        "stacked {:?}",
        Tensor::stack(&[&result, &result], 0).permute(&[1, 0, 2])
    );
    (colors - Tensor::stack(&[&result, &result], 0).permute(&[1, 0, 2]))
        .abs()
        .max()
        .print();
    // result.get(result.size1().unwrap() - 1).print();
    result
    // panic!();
    // .view((NUM_RAYS as i64, NUM_POINTS as i64, LABELS as i64))
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

fn compositing(densities: Tensor, colors: Tensor, distances: Tensor) -> Tensor {
    let tensor_vector: Vec<Tensor> = (0..NUM_POINTS)
        .map(|i| accumulated_transmittance(&densities, &distances, i as i64))
        .collect();

    let tensor_array: [Tensor; NUM_POINTS] = tensor_vector.try_into().unwrap();

    let T = Tensor::stack(&tensor_array, 0).view((NUM_RAYS as i64, NUM_POINTS as i64)); //TODO: check shaping/ordering

    let weights = (T * (1. as f32 - (-densities * distances).exp())).unsqueeze(2);
    // weights.softmax(1, Kind::Float).print();

    let final_colors: Tensor = (weights.softmax(1, Kind::Float) * colors).sum_dim_intlist(
        Some([1i64].as_slice()),
        false,
        Kind::Float,
    );

    return final_colors;
}

use tch::nn::ModuleT;
use vecmath::traits::Float;

pub struct TchModel {
    vs: nn::VarStore,
    net: Net,
    opt: Optimizer,
}

impl TchModel {
    pub fn new() -> TchModel {
        let vs = nn::VarStore::new(Device::Mps);
        let net = Net::new(&vs.root()); //net(&vs.root());
        let opt = nn::Adam::default().build(&vs, 5e-4).unwrap();

        TchModel { vs, net, opt }
    }

    pub fn predict(
        &self,
        coords: Vec<Vec<[f32; INDIM]>>,
        distances: Vec<[f32; NUM_POINTS]>,
    ) -> Tensor {
        const INDIM_BATCHED: usize = INDIM * BATCH_SIZE;
        let coords_flat =
            array_vec_to_1d_array::<INDIM, INDIM_BATCHED>(array_vec_vec_to_array_vec(coords));

        let coords_tensor = Tensor::of_slice(&coords_flat).view((BATCH_SIZE as i64, INDIM as i64));

        let densities_features = self.net.forward_t(&coords_tensor.to(Device::Mps), true);

        let densities = densities_features
            // .view((HIDDEN_NODES + 1 as i64, NUM_RAYS as i64, NUM_POINTS as i64))
            .view((NUM_RAYS as i64, NUM_POINTS as i64, HIDDEN_NODES + 1 as i64))
            .permute(&[2, 0, 1])
            .get(0);
        // .sigmoid();

        let features = densities_features
            .view((NUM_RAYS as i64, NUM_POINTS as i64, HIDDEN_NODES + 1 as i64))
            .slice(2 as i64, 1, HIDDEN_NODES + 1, 1);

        let colors = features.apply(&self.net.fc9).relu(); //.sigmoid(); // TODO: need batch first?

        // let distances_flat = array_vec_to_1d_array::<NUM_POINTS, BATCH_SIZE>(distances);
        // let mut distances_tensor =
        //     Tensor::of_slice(&distances_flat).view((NUM_RAYS as i64, NUM_POINTS as i64));

        // let tfar = Tensor::of_slice(&[10f32; NUM_RAYS]).unsqueeze(1);

        // distances_tensor = Tensor::concat(
        //     &[distances_tensor.slice(1, 1, NUM_POINTS as i64, 1), tfar], // TODO: check distances calculation
        //     1,
        // ) - distances_tensor;

        // compositing(densities, colors, distances_tensor.to(Device::Mps))
        select_compositing(colors)
    }

    pub fn step(&mut self, pred_tensor: Tensor, gold: Vec<[f32; LABELS]>) -> f32 {
        const LABELS_BATCHED: usize = LABELS * NUM_RAYS;
        let gold_flat = array_vec_to_1d_array::<LABELS, LABELS_BATCHED>(gold);
        let gold_tensor = Tensor::of_slice(&gold_flat).view((NUM_RAYS as i64, LABELS as i64));
        let loss = mse_loss(&pred_tensor, &gold_tensor.to(Device::Mps));
        // self.backward_scale_grad_step(&loss);
        self.opt.backward_step(&loss);

        return f32::try_from(&loss).unwrap();
    }

    fn backward_scale_grad_step(&mut self, loss: &Tensor) {
        self.opt.zero_grad();
        loss.backward();
        for var in self.vs.trainable_variables() {
            let mut grad = var.grad();
            grad *= NUM_POINTS as f32;
        }
        self.opt.step();
    }

    pub fn get_predictions_as_array_vec(&self, predictions: &Tensor) -> Vec<[f32; LABELS]> {
        tensor_to_array_vec(&predictions)
    }

    pub fn save(&self, save_path: &str) {
        self.vs.save(&save_path).unwrap();
    }

    pub fn load(&mut self, load_path: &str) {
        self.vs.load(&load_path).unwrap();
    }
}

fn array_vec_vec_to_array_vec(vv: Vec<Vec<[f32; INDIM]>>) -> Vec<[f32; INDIM]> {
    let mut v = Vec::new();
    for subvec in vv {
        for el in subvec {
            v.push(el);
        }
    }
    return v;
}

fn array_vec_to_1d_array<const INNER_DIM: usize, const OUT_DIM: usize>(
    v: Vec<[f32; INNER_DIM]>,
) -> [f32; OUT_DIM] {
    let mut array = [0f32; OUT_DIM];

    for batch_index in 0..NUM_RAYS {
        for item_index in 0..INNER_DIM {
            array[batch_index * INNER_DIM + item_index] = v[batch_index][item_index];
        }
    }
    return array;
}

pub fn tensor_to_array_vec(a: &Tensor) -> Vec<[f32; LABELS]> {
    let mut v = Vec::new();

    for i in 0..a.size()[0] {
        let mut r = [0f32; LABELS];
        for j in 0..LABELS {
            r[j] = a.double_value(&[i as i64, j as i64]) as f32;
        }
        v.push(r);
    }
    return v;
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}
