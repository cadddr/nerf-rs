use tch::{
    nn, nn::Module, nn::Optimizer, nn::OptimizerConfig, nn::Sequential, Device, Kind, Tensor,
};

pub const BATCH_SIZE: usize = 1024 * 64;

pub const INDIM: usize = 3;
const HIDDEN_NODES: i64 = 100;
const LABELS: usize = 4;

// MLP
fn net(vs: &nn::Path) -> Sequential {
    nn::seq()
        .add(nn::linear(
            vs / "layer1",
            INDIM as i64,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer2",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer3",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer4",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs / "layer5",
            HIDDEN_NODES,
            HIDDEN_NODES,
            Default::default(),
        ))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(
            vs,
            HIDDEN_NODES,
            LABELS as i64,
            Default::default(),
        ))
}

pub struct TchModel {
    vs: nn::VarStore,
    net: Sequential,
    opt: Optimizer,
}

impl TchModel {
    pub fn new() -> TchModel {
        let vs = nn::VarStore::new(Device::Mps);
        let net = net(&vs.root());
        let opt = nn::Adam::default().build(&vs, 5e-5).unwrap();

        TchModel { vs, net, opt }
    }

    pub fn predict(&self, coords: Vec<Vec<[f32; INDIM]>>) -> Tensor {
        const INDIM_BATCHED: usize = INDIM * BATCH_SIZE;
        let coords_flat =
            array_vec_to_1d_array::<INDIM, INDIM_BATCHED>(array_vec_vec_to_array_vec(coords));
        println!("INDIM_BATCHED {:?}", INDIM_BATCHED);
        let coords_tensor = Tensor::of_slice(&coords_flat).view((BATCH_SIZE as i64, INDIM as i64));
        // println!("{:?}", coords_tensor.size());
        let mut point_density_predictions = self.net.forward(&coords_tensor.to(Device::Mps));
        // println!("{:?}", point_density_predictions.size());
        point_density_predictions = point_density_predictions
            .view((1024 as i64, 64 as i64, LABELS as i64))
            .mean_dim(Some([1i64].as_slice()), false, Kind::Float);
        // println!("{:?}", point_density_predictions.size());
        return point_density_predictions;
    }

    pub fn step(&mut self, pred_tensor: Tensor, gold: Vec<[f32; LABELS]>) -> f32 {
        println!(
            "step pred_tensor {:?} gold {:?}",
            pred_tensor.size(),
            gold.len()
        );
        const LABELS_BATCHED: usize = LABELS * BATCH_SIZE / 64;
        let gold_flat = array_vec_to_1d_array::<LABELS, LABELS_BATCHED>(gold);
        println!("gold_flat {:?}", gold_flat.len());
        let gold_tensor =
            Tensor::of_slice(&gold_flat).view(((BATCH_SIZE / 64) as i64, LABELS as i64));
        println!("gold_tensor {:?}", gold_tensor.size());
        let loss = mse_loss(&pred_tensor, &gold_tensor.to(Device::Mps));
        self.opt.backward_step(&loss);

        return f32::try_from(&loss).unwrap();
    }

    pub fn BATCH_SIZE(&self) -> usize {
        BATCH_SIZE
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
    println!("vec vec {:?}", vv.len());
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
    println!("vec {:?}", v.len());
    let mut array = [0f32; OUT_DIM];
    println!("arr {:?}", array.len());

    for batch_index in 0..BATCH_SIZE / 64 {
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
