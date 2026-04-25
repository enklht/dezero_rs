use crate::functions::call_impl::call;
use crate::variable::VBox;

pub fn linear(x: &VBox, w: &VBox, b: Option<&VBox>) -> VBox {
    use crate::functions::Linear;

    let bias = b.is_some();
    let func = Linear::new(bias);
    if bias {
        call(func, &[x.clone(), w.clone(), b.unwrap().clone()])
    } else {
        call(func, &[x.clone(), w.clone()])
    }
}

pub fn sigmoid(x: &VBox) -> VBox {
    use crate::functions::Sigmoid;

    let func = Sigmoid::new();
    call(func, std::slice::from_ref(x))
}

pub fn relu(x: &VBox) -> VBox {
    use crate::functions::ReLU;

    let func = ReLU::new();
    call(func, std::slice::from_ref(x))
}

pub fn mean_squared_error(x: &VBox, y: &VBox) -> VBox {
    use crate::functions::MeanSquaredError;

    let func = MeanSquaredError::new();
    call(func, &[x.clone(), y.clone()])
}

pub fn softmax(x: &VBox, axis: usize) -> VBox {
    use crate::functions::Softmax;

    let func = Softmax::new(axis);
    call(func, std::slice::from_ref(x))
}

pub fn cross_entropy_loss(x: &VBox, t: &VBox) -> VBox {
    use crate::functions::CrossEntropy;

    let func = CrossEntropy::new();
    call(func, &[x.clone(), t.clone()])
}

pub fn softmax_cross_entropy_loss(x: &VBox, t: &VBox) -> VBox {
    use crate::functions::SoftmaxCrossEntropy;

    let func = SoftmaxCrossEntropy::new();
    call(func, &[x.clone(), t.clone()])
}
