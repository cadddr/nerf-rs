use tch::{Tensor, nn, nn::Module, nn::Optimizer, nn::OptimizerConfig, Device, Kind};

pub const BATCH_SIZE: usize = 128;

const IMAGE_DIM: i64 = 2;
const HIDDEN_NODES: i64 = 100;
const LABELS: i64 = 4;

// MLP
fn net(vs: &nn::Path) -> impl Module {
    nn::seq()
        .add(nn::linear(vs / "layer1", IMAGE_DIM, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "layer2", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "layer3", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "layer4", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs / "layer5", HIDDEN_NODES, HIDDEN_NODES, Default::default()))
        .add_fn(|xs| xs.tanh())
        .add(nn::linear(vs, HIDDEN_NODES, LABELS, Default::default()))
}
// pub fn init_mlp() -> (MLP, Adam<MLP>) {
pub fn init_mlp() -> (impl Module, Optimizer){
    let vs = nn::VarStore::new(Device::Mps);
    let net = net(&vs.root());
    let mut opt = nn::Adam::default().build(&vs, 5e-5).unwrap();
	
	return (net, opt)
}
	
// pub fn step(model: &mut MLP, opt: &mut Adam<MLP>, y: Tensor2D<BATCH_SIZE, 4, OwnedTape>, gold: Vec<[f32; 4]>) -> f32 {
	
pub fn step(net: impl Module, opt: &mut Optimizer, pred_tensor: Tensor, gold: Vec<[f32; 4]>) -> f32{
	// let m = tch::vision::mnist::load_dir("data")?;
    // let p = net.forward(&m.train_images.to(Device::Mps));
	const dim: usize = 4 * BATCH_SIZE;
	let gold_tensor = Tensor::of_slice(&array_vec_to_1d_array::<4, dim>(gold)).view((i64::from(BATCH_SIZE as i32), i64::from(4)));
	let loss = mse_loss(&pred_tensor, &gold_tensor.to(Device::Mps));
    opt.backward_step(&loss);
	
	return f32::try_from(&loss).unwrap();
}
	
// pub fn predict_emittance_and_density(mlp: &MLP, coords: Vec<[f32; 2]>, views: Vec<[f32; 3]>, points: Vec<[f32; 3]>) -> Tensor2D<BATCH_SIZE, 4, OwnedTape> {
pub fn predict(net: impl Module, coords: Vec<[f32; 2]>) -> Tensor {
	const dim: usize = 2 * BATCH_SIZE;
	let coords_tensor = Tensor::of_slice(&array_vec_to_1d_array::<2, dim>(coords)).view((i64::from(BATCH_SIZE as i32), i64::from(2)));
	let p = net.forward(&coords_tensor.to(Device::Mps));
	return p;
}

fn array_vec_to_1d_array<const D:usize, const OUT:usize>(v: Vec<[f32; D]>) -> [f32; OUT] {
	let mut array = [0f32; OUT];
	
    for batch_index in 0..BATCH_SIZE {
        for item_index in 0..D {
			array[batch_index * D * BATCH_SIZE + item_index] = v[batch_index][item_index];
		}
	}
	return array;
}

fn mse_loss(x: &Tensor, y: &Tensor) -> Tensor {
    let diff = x - y;
    (&diff * &diff).mean(Kind::Float)
}
