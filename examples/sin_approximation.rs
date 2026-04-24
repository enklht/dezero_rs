use std::f32::consts::PI;

use dezero::functions as F;
use dezero::layers::{Model, MLP};
use dezero::optimizers::{Momentum, Optimizer};
use dezero::{array::Array, array1, var};

fn main() {
    let x_data = (array1!(0..100) / 100.).reshape(&[100, 1]);
    let y_data = (2. * PI * &x_data).sin() + Array::randn(&[100, 1], 0., 0.01);
    let x = var!(x_data);
    let y = var!(y_data);

    let model = Model::new(MLP::new(&[10, 1], Box::new(F::sigmoid)));

    let mut optimizer = Momentum::new(0.05, 0.9, model.clone());

    let iters = 3000;
    for i in 0..iters {
        let y_pred = model.call(x);
        let loss = F::mean_squared_error(y, &y_pred);

        model.clear_grads();
        loss.backward();
        optimizer.update();

        if i % 300 == 0 {
            println!("{loss}");
        }
    }

    dezero::eval!();
    model
        .call(x)
        .get_array()
        .write_csv("data/sin_regression_pred.csv");
    y.get_array().write_csv("data/sin_regression_target.csv");
}
