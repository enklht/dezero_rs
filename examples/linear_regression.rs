use dezero::functions as F;
use dezero::layers::{Model, MLP};
use dezero::optimizers::{Momentum, Optimizer};
use dezero::{array::Array, array1, var};

fn main() {
    let x_data = (array1!(0..100) / 100.).reshape(&[100, 1]);
    let y_data = 3. * &x_data + 2. + Array::randn(&[100, 1], 0., 0.01);

    let x = var!(x_data);
    let y = var!(y_data);

    let model = Model::new(MLP::new(&[1], Box::new(F::relu)));
    let mut optimizer = Momentum::new(0.01, 0.9, model.clone());

    let iters = 2000;
    for i in 0..iters {
        let y_pred = model.call(x);
        let loss = F::mean_squared_error(y, &y_pred);

        model.clear_grads();
        loss.backward();
        optimizer.update();

        if i % 200 == 0 {
            println!("{loss}");
        }
    }

    dezero::eval!();
    model
        .call(x)
        .get_array()
        .write_csv("data/linear_regression_pred.csv");
    y.get_array().write_csv("data/linear_regression_target.csv");
}
