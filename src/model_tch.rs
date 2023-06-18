use tch::{Tensor, nn, nn::Module, nn::Optimizer, nn::OptimizerConfig, Device};

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
	Tensor::from_slice(&[10f32, 20f32]);
	let gold_tensor = Tensor::from_slice(&[10f32, 20f32]);//(&gold[..]).unwrap();
	let loss = pred_tensor.cross_entropy_for_logits(&gold_tensor.to(Device::Mps));
    opt.backward_step(&loss);
	
	return f32::try_from(&loss).unwrap();
}
	
// pub fn predict_emittance_and_density(mlp: &MLP, coords: Vec<[f32; 2]>, views: Vec<[f32; 3]>, points: Vec<[f32; 3]>) -> Tensor2D<BATCH_SIZE, 4, OwnedTape> {

