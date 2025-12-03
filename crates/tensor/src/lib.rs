use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;

// Type alias for shared data buffer
type SharedData<T> = Rc<RefCell<Vec<T>>>;

// Placeholder for the actual gradient computation logic
pub trait GradFunction<T> {
    fn backward(
        &self,
        grad_output: &Tensor<T, impl AutoGrad, impl AutoGrad>,
    ) -> Vec<(
        Tensor<T, impl AutoGrad, impl AutoGrad>,
        Tensor<T, impl AutoGrad, impl AutoGrad>,
    )>;
}

// Trait to define the common interface for Autograd state (G in Tensor<...>)
pub trait AutoGrad {
    // Compile-time constant to check if gradient tracking is required
    const REQUIRES_GRAD: bool;

    // Allows accessing grad_fn safely, returns None for NoExtraInfo
    fn get_grad_fn(&self) -> Option<&Rc<dyn GradFunction<f32>>>;
}

// Marker struct for Tensors that DO NOT track gradients (Zero Sized Type)
pub struct NoExtraInfo;

impl AutoGrad for NoExtraInfo {
    const REQUIRES_GRAD: bool = false;

    fn get_grad_fn(&self) -> Option<&Rc<dyn GradFunction<f32>>> {
        None // No gradient function exists
    }
}

// Struct for Tensors that DO track gradients (always holds data)
pub struct AutogradInfo<T> {
    pub grad_fn: Rc<dyn GradFunction<T>>,
    // PhantomData is used here to satisfy type constraints
    // when T is used in the associated GradFunction
    _phantom: PhantomData<T>,
}

impl AutoGrad for AutogradInfo<f32> {
    const REQUIRES_GRAD: bool = true;

    fn get_grad_fn(&self) -> Option<&Rc<dyn GradFunction<f32>>> {
        // Data is present
        Some(&self.grad_fn)
    }
}

// Core required metadata
pub struct BaseTensor<T> {
    pub data: SharedData<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

// The final composed Tensor struct with multiple generic features
// G must implement AutoGrad. Q is a placeholder for future features (e.g., Quantization)
pub struct Tensor<T, G: AutoGrad, Q> {
    pub base: BaseTensor<T>,
    pub grad: G,
    pub quant: Q, // Placeholder for future features
}

// Tensor type used for Training (with Autograd tracking)
pub type TrainingTensor<T> = Tensor<T, AutogradInfo<T>, NoExtraInfo>;

// Tensor type used for Inference (without Autograd tracking)
pub type InferenceTensor<T> = Tensor<T, NoExtraInfo, NoExtraInfo>;

impl<T, G: AutoGrad, Q> Tensor<T, G, Q> {
    // A method that utilizes the AutoGrad trait boundary
    pub fn add(&self, other: &Self) -> Self {
        // ... (performs element-wise addition)

        // Autograd logic check
        if G::REQUIRES_GRAD {
            // This entire block is optimized away if G is NoExtraInfo
            // Create a new grad_fn for the addition operation
            // and wrap the result with AutogradInfo

            // NOTE: Returning Self requires logic to construct the new AutogradInfo
            // or NoExtraInfo based on the inputs' G type, which complicates the example.
            // For simplicity, assume the caller handles the type transition.
            println!("INFO: Gradient tracking is active.");
        } else {
            println!("INFO: Gradient tracking is skipped.");
        }

        // Dummy return for structural completeness
        Self {
            base: self.base.clone(),                      // Simplified copy
            grad: G::get_new_state_after_op(self, other), // Requires complex trait extension
            quant: self.quant,                            // Simplified copy
        }
    }
}
