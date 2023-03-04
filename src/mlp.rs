use dfdx::tensor::{tensor, Tensor, Tensor1D, Tensor2D, TensorCreator};
use dfdx::gradients::{CanUpdateWithGradients, GradientProvider, OwnedTape, Tape, UnusedTensors};
use dfdx::nn::{Linear, Module, ReLU, ResetParams};

#[derive(Default)]
pub struct Mlp<const IN: usize, const INNER: usize, const OUT: usize> {
    l1: Linear<IN, INNER>,
    l2: Linear<INNER, OUT>,
    relu: ReLU,
}

// ResetParams lets you randomize a model's parameters
impl<const IN: usize, const INNER: usize, const OUT: usize> ResetParams for Mlp<IN, INNER, OUT> {
    fn reset_params<R: rand::Rng>(&mut self, rng: &mut R) {
        self.l1.reset_params(rng);
        self.l2.reset_params(rng);
        self.relu.reset_params(rng);
    }
}

// CanUpdateWithGradients lets you update a model's parameters using gradients
impl<const IN: usize, const INNER: usize, const OUT: usize> CanUpdateWithGradients
for Mlp<IN, INNER, OUT>
{
    fn update<G: GradientProvider>(&mut self, grads: &mut G, unused: &mut UnusedTensors) {
        self.l1.update(grads, unused);
        self.l2.update(grads, unused);
        self.relu.update(grads, unused);
    }
}

// impl Module for single item
impl<const IN: usize, const INNER: usize, const OUT: usize> Module<Tensor1D<IN>>
for Mlp<IN, INNER, OUT>
{
    type Output = Tensor1D<OUT>;

    fn forward(&self, x: Tensor1D<IN>) -> Self::Output {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}

// impl Module for batch of items
impl<const BATCH: usize, const IN: usize, const INNER: usize, const OUT: usize, TAPE: Tape>
Module<Tensor2D<BATCH, IN, TAPE>> for Mlp<IN, INNER, OUT>
{
    type Output = Tensor2D<BATCH, OUT, TAPE>;

    fn forward(&self, x: Tensor2D<BATCH, IN, TAPE>) -> Self::Output {
        let x = self.l1.forward(x);
        let x = self.relu.forward(x);
        self.l2.forward(x)
    }
}