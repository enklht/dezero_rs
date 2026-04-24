extern crate dezero;

use dezero::functions::{self as F, mean_squared_error};
use dezero::{array0, array1, array2, array_with_shape, scaler, var, variable::VBox};

macro_rules! square {
    ($x: expr) => {
        $x.clone() * $x.clone()
    };
}

#[test]
fn square_backward() {
    let x = scaler!(3.);
    let y = square!(x);
    y.backward();
    assert_eq!(x.get_grad(), array0!(6));
}

#[test]
fn add() {
    let x0 = scaler!(2.);
    let x1 = scaler!(3.);
    let y = x0 + x1;
    assert_eq!(y.get_array(), array0!(5));
}

#[test]
fn square_add() {
    let x = scaler!(2.);
    let y = scaler!(3.);
    let z = square!(x) + square!(y);
    z.backward();
    assert_eq!(z.get_array(), array0!(13));
    assert_eq!(x.get_grad(), array0!(4));
    assert_eq!(y.get_grad(), array0!(6));
}

#[test]
fn add_backward() {
    let x = scaler!(3.);
    let y = x + x;
    y.backward();
    assert_eq!(x.get_grad(), array0!(2));
}

#[test]
fn clear_grad() {
    let x = scaler!(3.);
    let y = x + x;
    y.backward();
    assert_eq!(x.get_grad(), array0!(2.));

    x.clear_grad();
    let y = x + x + x;
    y.backward();
    assert_eq!(x.get_grad(), array0!(3.));
}

#[test]
fn complex_graph() {
    let x = scaler!(2.);
    let a = square!(x);
    let y = square!(a) + square!(a);
    y.backward();

    assert_eq!(y.get_array(), array0!(32.));
    assert_eq!(x.get_grad(), array0!(64.));
}

#[test]
fn grad_drop() {
    let x0 = scaler!(1.);
    let x1 = scaler!(1.);

    let t = x0 + x1;
    let y = x0 + &t;

    y.backward();

    assert_eq!(y.get_option_grad(), None);
    assert_eq!(t.get_option_grad(), None);
    assert_eq!(x0.get_option_grad(), Some(array0!(2.)));
    assert_eq!(x1.get_option_grad(), Some(array0!(1.)));
}

#[test]
fn overload() {
    let a = scaler!(3.);
    let b = scaler!(2.);
    let c = scaler!(1.);

    let y = a * b + c;
    y.backward();

    assert_eq!(y.get_array(), array0!(7.));
    assert_eq!(a.get_grad(), array0!(2.));
    assert_eq!(b.get_grad(), array0!(3.));
}

#[test]
fn sphere() {
    let x = scaler!(1.);
    let y = scaler!(1.);
    let z = x.powi(2) + y.powi(2);

    z.backward();

    assert_eq!(x.get_grad(), array0!(2.));
    assert_eq!(y.get_grad(), array0!(2.));
}

#[test]
fn matyas() {
    let matyas = |x: &VBox, y: &VBox| 0.26 * (x.powi(2) + y.powi(2)) - 0.48 * x * y;

    let x = &scaler!(1.);
    let y = &scaler!(1.);
    let z = matyas(x, y);
    z.backward();

    assert!(x.get_grad().all_close(&array0!(0.04), 1e-8));
    assert!(y.get_grad().all_close(&array0!(0.04), 1e-8));
}

#[test]
fn goldstein_prince() {
    let gp = |x: &VBox, y: &VBox| {
        (1 + (x + y + 1).powi(2)
            * (19 - 14 * x + 3 * x.powi(2) - 14 * y + 6 * x * y + 3 * y.powi(2)))
            * (30
                + (2 * x - 3 * y).powi(2)
                    * (18 - 32 * x + 12 * x.powi(2) + 48 * y - 36 * x * y + 27 * y.powi(2)))
    };

    let x = &scaler!(1);
    let y = &scaler!(1);
    let z = gp(x, y);
    z.backward();

    assert!(x.get_grad().all_close(&array0!(-5376.), 1e-8));
    assert!(y.get_grad().all_close(&array0!(8064), 1e-8));
}

#[test]
fn rosenbrock_opt() {
    let rosenbrock = |x0: &VBox, x1: &VBox| 100 * (x1 - x0.powi(2)).powi(2) + (x0 - 1).powi(2);

    let x0 = scaler!(0);
    let x1 = scaler!(2);

    let lr = 0.001;
    let max_iter = 20000;

    for _ in 0..max_iter {
        let y = rosenbrock(x0, x1);

        x0.clear_grad();
        x1.clear_grad();
        y.backward();

        x0.set_array(x0.get_array() - lr * x0.get_grad());
        x1.set_array(x1.get_array() - lr * x1.get_grad());
    }

    assert!(x0.get_array().all_close(&array0!(1), 1e-3));
    assert!(x1.get_array().all_close(&array0!(1), 1e-3));
}

#[test]
fn reshape() {
    let x = var!(array_with_shape!(0..6, [2, 3]));
    let y = x.reshape(vec![6]);
    y.backward();
    assert_eq!(x.get_grad(), array_with_shape!([1; 6], [2, 3]));
    assert_eq!(y.get_array(), array_with_shape!(0..6, [6]))
}

#[test]
fn transpose() {
    let x = var!(array2!([[1, 2, 3], [4, 5, 6]]));
    let y = x.transpose();
    y.backward();
    assert_eq!(x.get_grad(), array_with_shape!([1; 6], [2, 3]));
    assert_eq!(y.get_array(), array_with_shape!([1, 4, 2, 5, 3, 6], [3, 2]))
}

#[test]
fn linear() {
    let x = var!(array1!(0..12).reshape(&[3, 4]));
    let y = var!(array1!(0..16).reshape(&[4, 4]));
    let b = var!(array1!(0..4));
    let z = &F::linear(x, y, Some(b));
    z.backward();

    dbg!(array1!(0..4).broadcast_to(&[3, 4]));

    assert_eq!(
        z.get_array(),
        array2!([[56, 62, 68, 74], [152, 174, 196, 218], [248, 286, 324, 362]])
            + array2!([[0, 1, 2, 3]; 3])
    );
    assert_eq!(x.get_grad(), array2!([[6, 22, 38, 54]; 3]));
    assert_eq!(y.get_grad(), array2!([[12; 4], [15; 4], [18; 4], [21; 4]]));
    dbg!(b.get_grad());
}

#[test]
fn mse() {
    let x = var!(array1!(0..5));
    let y = var!(array1!(5..10));

    let z = &mean_squared_error(x, y);
    z.backward();

    assert_eq!(array0!(25), z.get_array());
    assert_eq!(array1!([-2; 5]), x.get_grad());
    assert_eq!(array1!([2; 5]), y.get_grad());
}

#[test]
fn sigmoid() {
    let x = scaler!(0.5);
    let x = &x.broadcast_to(&[10, 1]);
    println!("{}", x);
    let y = &F::sigmoid(x);

    y.backward();
}

#[test]
fn broadcast() {
    let x = array1!(0..2);
    assert_eq!(x.broadcast_to(&[2, 2]), array2!([[0, 1], [0, 1]]));
}

#[test]
fn backward() {
    let x = &VBox::new(array1!(0..10));
    let y = &F::softmax(x, 0);

    let t = &VBox::new(array1!([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]));

    let loss = &F::cross_entropy_loss(y, t);
    loss.backward();
    println!("{}", x);

    let delta = 1e-3;
    let d_x = &VBox::new(array1!([0., 0., 0., 0., 0., 0., 0., 0., 0., delta]));
    let d_loss = &F::cross_entropy_loss(&F::softmax(&(x + d_x), 0), t);
    println!("{}", (d_loss.get_array() - loss.get_array()) / delta);

    // panic!()
}

#[test]
fn all_close_handles_negative_diff() {
    let x = array1!([1.0, 2.0]);
    let y = array1!([1.05, 2.05]);

    assert!(!x.all_close(&y, 0.01));
    assert!(x.all_close(&y, 0.1));
}

#[test]
fn broadcasted_matmul_backward_shapes() {
    let x = var!(array_with_shape!(0..8, [2, 1, 2, 2]));
    let w = var!(array_with_shape!(0..12, [1, 3, 2, 2]));

    let y = x.matmul(w);
    let loss = y.sum();
    loss.backward();

    assert_eq!(x.get_grad().get_shape(), &[2, 1, 2, 2]);
    assert_eq!(w.get_grad().get_shape(), &[1, 3, 2, 2]);
}

#[test]
fn broadcasted_linear_backward_shapes() {
    let x = var!(array_with_shape!(0..6, [2, 1, 3]));
    let w = var!(array_with_shape!(0..12, [1, 3, 4]));
    let b = var!(array1!(0..4));

    let y = F::linear(x, w, Some(b));
    let loss = y.sum();
    loss.backward();

    assert_eq!(x.get_grad().get_shape(), &[2, 1, 3]);
    assert_eq!(w.get_grad().get_shape(), &[1, 3, 4]);
    assert_eq!(b.get_grad().get_shape(), &[4]);
}
