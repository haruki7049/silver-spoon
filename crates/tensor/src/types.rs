use std::rc::Rc;
use std::cell::RefCell;

type SharedData<T> = Rc<RefCell<Vec<T>>>;

#[derive(Clone, PartialEq, Eq)]
pub struct BaseTensor<T> {
    pub data: SharedData<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Tensor<T> {
    pub base: BaseTensor<T>,
}

impl<T> std::ops::Add for BaseTensor<T>
where
    T: std::ops::Add<Output = T> + Copy + Default,
{
    type Output = BaseTensor<T>;

    fn add(self, rhs: Self) -> Self::Output {
        // 1. Check if shapes match
        if self.shape != rhs.shape {
            // In a real implementation, this should panic or return a Result.
            panic!("BaseTensor shapes must match for addition: left {:?} vs right {:?}", self.shape, rhs.shape);
        }

        let total_size = self.shape.iter().product();

        // 2. Create a new Vec to store the result
        let mut result_data = Vec::with_capacity(total_size);

        // Get shared data references. `borrow()` is needed for RefCell.
        let left_data = self.data.borrow();
        let right_data = rhs.data.borrow();

        // Simplified addition assuming contiguous and simple data layout.
        // A full implementation would need an iterator that respects shape, strides, and offset.
        for i in 0..total_size {
            // Element-wise addition. We need to use `T::add` or the `+` operator.
            // Since T: Add<Output = T>, we can use the + operator.
            // Using `*` to dereference the Copy type and apply the operation.
            let sum = left_data[i].add(right_data[i]);
            result_data.push(sum);
        }

        // 3. Construct the result BaseTensor
        BaseTensor {
            // Wrap the new data in Rc<RefCell<...>> for the result
            data: Rc::new(RefCell::new(result_data)),
            // The shape, strides, and offset are the same as the inputs (for a simple view)
            shape: self.shape,
            strides: self.strides,
            offset: 0, // New data starts at offset 0
        }
    }
}

impl<T> std::ops::Add for Tensor<T>
where
    // T must support addition and its output must be T
    T: std::ops::Add<Output = T> + Copy + Default,
    // Note: The specific trait bounds depend on the actual needs.
    // Copy is needed for element-wise operation, Default is for creating the new Vec.
{
    // The type produced in the output
    type Output = Tensor<T>;

    // The method to perform the addition
    fn add(self, rhs: Self) -> Self::Output {
        Tensor {
            base: self.base + rhs.base,
        }
    }
}

impl<T> std::ops::Sub for Tensor<T>
where
    // T must support addition and its output must be T
    T: std::ops::Sub<Output = T> + Copy + Default,
    // Note: The specific trait bounds depend on the actual needs.
    // Copy is needed for element-wise operation, Default is for creating the new Vec.
{
    // The type produced in the output
    type Output = Tensor<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        // 1. Check if shapes match
        if self.base.shape != rhs.base.shape {
            // In a real implementation, this should panic or return a Result.
            panic!("Tensor shapes must match for addition: left {:?} vs right {:?}", self.base.shape, rhs.base.shape);
        }

        let total_size = self.base.shape.iter().product();

        // 2. Create a new Vec to store the result
        let mut result_data = Vec::with_capacity(total_size);

        // Get shared data references. `borrow()` is needed for RefCell.
        let left_data = self.base.data.borrow();
        let right_data = rhs.base.data.borrow();

        for i in 0..total_size {
            let sum = left_data[i].sub(right_data[i]);
            result_data.push(sum);
        }

        // 3. Construct the result Tensor
        Tensor {
            base: BaseTensor {
                // Wrap the new data in Rc<RefCell<...>> for the result
                data: Rc::new(RefCell::new(result_data)),
                // The shape, strides, and offset are the same as the inputs (for a simple view)
                shape: self.base.shape,
                strides: self.base.strides,
                offset: 0, // New data starts at offset 0
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Tensor, BaseTensor};
    use std::rc::Rc;
    use std::cell::RefCell;

    #[test]
    fn add() {
        let left = Tensor {
            base: BaseTensor {
                data: Rc::new(RefCell::new(vec![1, 2, 3, 4])),
                shape: vec![2, 2],
                strides: vec![2, 1],
                offset: 0,
            },
        };
        let right = Tensor {
            base: BaseTensor {
                data: Rc::new(RefCell::new(vec![10, 20, 30, 40])),
                shape: vec![2, 2],
                strides: vec![2, 1],
                offset: 0,
            },
        };

        let result = left + right;

        assert_eq!(*result.base.data.borrow(), vec![11, 22, 33, 44]);
    }

    #[test]
    fn sub() {
        let left = Tensor {
            base: BaseTensor {
                data: Rc::new(RefCell::new(vec![10, 20, 30, 40])),
                shape: vec![2, 2],
                strides: vec![2, 1],
                offset: 0,
            },
        };
        let right = Tensor {
            base: BaseTensor {
                data: Rc::new(RefCell::new(vec![1, 2, 3, 4])),
                shape: vec![2, 2],
                strides: vec![2, 1],
                offset: 0,
            },
        };

        let result = left - right;

        assert_eq!(*result.base.data.borrow(), vec![9, 18, 27, 36]);
    }

    #[test]
    fn sub_with_minus_result() {
        let left = Tensor {
            base: BaseTensor {
                data: Rc::new(RefCell::new(vec![1, 2, 3, 4])),
                shape: vec![2, 2],
                strides: vec![2, 1],
                offset: 0,
            },
        };
        let right = Tensor {
            base: BaseTensor {
                data: Rc::new(RefCell::new(vec![10, 20, 30, 40])),
                shape: vec![2, 2],
                strides: vec![2, 1],
                offset: 0,
            },
        };

        let result = left - right;

        assert_eq!(*result.base.data.borrow(), vec![-9, -18, -27, -36]);
    }
}
