use tch::{Tensor, nn, nn::Module, nn::Sequential, nn::Optimizer, nn::OptimizerConfig, Device, Kind};

pub const BATCH_SIZE: usize = 4096;

pub const INDIM: usize = 6;
const HIDDEN_NODES: i64 = 100;
const LABELS: i64 = 4;

// MLP
fn net(vs: &nn::Path) -> Sequential {
    nn::seq()
        .add(nn::linear(vs / "layer1", INDIM as i64, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer2", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer3", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer4", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs / "layer5", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.relu())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}

pub struct TchModel {
	net: Sequential,
	opt: Optimizer
}

impl TchModel {
	pub fn new() -> TchModel {
		let (net, opt) = init_mlp();
		TchModel {
			net: net,
			opt: opt
		}
	}
	
	pub fn step(&mut self, pred_tensor: Tensor, gold: Vec<[f32; 4]>) -> f32 {
		step(&self.net, &mut self.opt, pred_tensor, gold)
	}
	
	pub fn predict(&self, coords: Vec<[f32; INDIM]>, views: Vec<[f32; 3]>, points: Vec<[f32; 3]>) -> Tensor {
		predict(&self.net, coords, views, points)
	}
	
	pub fn BATCH_SIZE(&self) -> usize {
		BATCH_SIZE
	}
	
	pub fn get_predictions_as_array_vec(&self, predictions: &Tensor) -> Vec<[f32; 4]> {
		tensor_to_vec(&predictions)
	}
}


pub fn init_mlp() -> (Sequential, Optimizer){
    let vs = nn::VarStore::new(Device::Mps);
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 1e-3).unwrap();
	
	return (net, opt)
}

	
pub fn step(net: &impl Module, opt: &mut Optimizer, pred_tensor: Tensor, gold: Vec<[f32; 4]>) -> f32{
	const dim: usize = 4 * BATCH_SIZE;
	let gold_tensor = Tensor::of_slice(&array_vec_to_1d_array::<4, dim>(gold)).view((i64::from(BATCH_SIZE as i32), i64::from(4)));
	let loss = mse_loss(&pred_tensor, &gold_tensor.to(Device::Mps));
    opt.backward_step(&loss);
	
	return f32::try_from(&loss).unwrap();
}
	
pub fn predict(net: &impl Module, coords: Vec<[f32; INDIM]>, views: Vec<[f32; 3]>, points: Vec<[f32; 3]>) -> Tensor {
	const dim: usize = INDIM * BATCH_SIZE;
	let coords_tensor = Tensor::of_slice(&array_vec_to_1d_array::<INDIM, dim>(coords)).view((i64::from(BATCH_SIZE as i32), INDIM as i64));
	let mut p = net.forward(&coords_tensor.to(Device::Mps));
	return p;
}

fn array_vec_to_1d_array<const D:usize, const OUT:usize>(v: Vec<[f32; D]>) -> [f32; OUT] {
	let mut array = [0f32; OUT];

    for batch_index in 0..BATCH_SIZE {
        for item_index in 0..D {
			array[batch_index * D + item_index] = v[batch_index][item_index];
		}
	}
	return array;
}

pub fn tensor_to_vec(a: &Tensor) -> Vec<[f32; 4]> {
	let mut v = Vec::new();
	
	for i in 0..a.size()[0] {
		let mut r = [0f32; 4];
		for j in 0..4 {
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
