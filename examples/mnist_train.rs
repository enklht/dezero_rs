use dezero::{
    array::Array,
    functions as F,
    layers::{Model, MLP},
    optimizers::{Momentum, Optimizer},
    variable::VBox,
};

fn main() {
    let (vec_x, vec_t) = load_mnist("data/mnist_test.csv");

    let model = Model::new(MLP::new(&[100, 10], Box::new(F::relu)));

    let mut optimizer = Momentum::new(0.1, 0.9, model.clone());

    let epochs = 10;

    for i in 0..epochs {
        let mut loss_tot = 0.;
        for (x, t) in vec_x.iter().zip(vec_t.iter()) {
            let x = VBox::new(x.clone());
            let t = VBox::new(t.clone());
            let y = F::softmax(&model.call(&x), 1);
            let loss = F::cross_entropy_loss(&y, &t);

            loss_tot += loss.get_array().get_data()[0];

            model.clear_grads();
            loss.backward();

            optimizer.update();
        }
        println!("epoch {i}");
        println!("loss: {loss_tot}");
    }

    dezero::eval!();
    let x = VBox::new(vec_x[0].clone());
    let y = F::softmax(&model.call(&x), 1);
    y.get_array().write_csv("data/mnist_res_test.csv");
}

fn load_mnist(path: &str) -> (Vec<Array>, Vec<Array>) {
    println!("loading...");
    let f = std::fs::read_to_string(path).expect("File not found");
    let lines = f.lines().collect::<Vec<_>>();
    let num_cols = lines[0].split(',').count() - 1;

    let mut vec_data = Vec::new();
    let mut vec_target = Vec::new();

    for minibatch in lines.chunks(100) {
        let num_rows = minibatch.len();
        let mut data = Vec::new();
        let mut target = Vec::new();
        for line in minibatch {
            let line = line
                .split(',')
                .map(|d| d.parse::<f32>().unwrap())
                .collect::<Vec<_>>();
            let mut target_row = vec![0.; 10];
            target_row[line[0] as usize] = 1.;
            data.append(&mut line[1..].to_vec());
            target.append(&mut target_row);
        }
        let x = Array::new(data, vec![num_rows, num_cols]);
        let t = Array::new(target, vec![num_rows, 10]);
        vec_data.push(x);
        vec_target.push(t);
    }

    println!("load finished");
    (vec_data, vec_target)
}
