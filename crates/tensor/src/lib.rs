use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

// Type alias for shared data buffer
type SharedData<T> = Rc<RefCell<Vec<T>>>;

// Placeholder for the actual gradient computation logic
// Note: Removed the generic T and used the concrete TrainingTensor<f32>
// to ensure the trait is object safe (dyn compatible).
pub trait GradFunction {
    // We use a concrete type alias here to avoid the non-object-safe impl Trait
    fn backward(&self, grad_output: &TrainingTensor<f32>) -> Vec<TrainingTensor<f32>>;
}

// Trait to define the common interface for Autograd state (G in Tensor<...>)
pub trait AutoGrad: Clone {
    // Added Clone trait bound
    // Compile-time constant to check if gradient tracking is required
    const REQUIRES_GRAD: bool;

    // Allows accessing grad_fn safely, returns None for NoExtraInfo
    // Note: Removed the generic <f32> from dyn GradFunction
    fn get_grad_fn(&self) -> Option<&Rc<dyn GradFunction>>;
}

// Marker struct for Tensors that DO NOT track gradients (Zero Sized Type)
#[derive(Clone)] // Added Clone derive
pub struct NoExtraInfo;

impl AutoGrad for NoExtraInfo {
    const REQUIRES_GRAD: bool = false;

    fn get_grad_fn(&self) -> Option<&Rc<dyn GradFunction>> {
        None // No gradient function exists
    }
}

// Struct for Tensors that DO track gradients (always holds data)
#[derive(Clone)] // Added Clone derive
pub struct AutogradInfo<T> {
    // Note: Removed the generic <T> from dyn GradFunction
    pub grad_fn: Rc<dyn GradFunction>,
    // PhantomData is used here to satisfy type constraints
    _phantom: PhantomData<T>,
}

impl AutoGrad for AutogradInfo<f32> {
    const REQUIRES_GRAD: bool = true;

    fn get_grad_fn(&self) -> Option<&Rc<dyn GradFunction>> {
        // Data is present
        Some(&self.grad_fn)
    }
}

// Core required metadata
#[derive(Clone)] // Added Clone derive to enable self.base.clone()
pub struct BaseTensor<T> {
    pub data: SharedData<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

// The final composed Tensor struct with multiple generic features
pub struct Tensor<T, G: AutoGrad, Q: Clone> {
    // Added Clone bound to Q
    pub base: BaseTensor<T>,
    pub grad: G,
    pub quant: Q, // Placeholder for future features
}

// Tensor type used for Training (with Autograd tracking)
pub type TrainingTensor<T> = Tensor<T, AutogradInfo<T>, NoExtraInfo>;

// Tensor type used for Inference (without Autograd tracking)
pub type InferenceTensor<T> = Tensor<T, NoExtraInfo, NoExtraInfo>;

impl<T, G: AutoGrad, Q: Clone> Tensor<T, G, Q> {
    // A method that utilizes the AutoGrad trait boundary
    pub fn add(&self, other: &Self) -> Self
    where
        T: Clone, // Required if T is used in BaseTensor operations, or if BaseTensor has T fields.
    {
        // ... (performs element-wise addition)

        // Autograd logic check
        if G::REQUIRES_GRAD {
            println!("INFO: Gradient tracking is active.");
        } else {
            println!("INFO: Gradient tracking is skipped.");
        }

        // Return new Tensor. We clone the necessary components.
        Self {
            base: self.base.clone(),
            // Fixed the placeholder: clone the existing grad state (G implements Clone via AutoGrad)
            grad: self.grad.clone(),
            // Fixed the placeholder: clone the quant state (Q implements Clone)
            quant: self.quant.clone(),
        }
    }
}
