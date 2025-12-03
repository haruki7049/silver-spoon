// Type alias for shared data buffer
type SharedData<T> = Rc<RefCell<Vec<T>>>;

#[derive(Clone)]
pub struct BaseTensor<T> {
    pub data: SharedData<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

pub struct Tensor<T> {
    pub base: BaseTensor<T>,
}
