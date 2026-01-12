use std::f32::consts::PI;

use dezero::functions as F;
use dezero::layers::{Model, MLP};
use dezero::optimizers::{Momentum, Optimizer};
use dezero::{array::Array, array1, var};

fn main() {
    let x = (array1!(0..100) / 100.).reshape(&[100, 1]);
    let y = (2. * PI * &x).sin() + Array::randn(&[100, 1], 0., 0.05);
    let x = var!(x.reshape(&[100, 1]));
    let y = var!(y);

    let model = Model::new(MLP::new(&[10, 10, 1], Box::new(F::relu)));

    let mut optimizer = Momentum::new(0.0002, 0.9, model.clone());

    let iters = 10000;

    for i in 0..iters {
        let y_pred = model.call(x);
        let loss = F::mean_squared_error(y, &y_pred);

        model.clear_grads();
        loss.backward();

        optimizer.update();

        if i % 1000 == 0 {
            println!("{loss}");
        }
    }
}
