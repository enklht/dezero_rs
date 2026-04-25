use crate::array::Array;
use crate::functions::macros::*;
use crate::functions::trait_::Function;
use crate::variable::{VBox, WeakVBox};

define!(Linear, bias: bool);

impl Function for Linear {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        let t = x[0].matmul(&x[1]);
        if self.bias {
            t + &x[2]
        } else {
            t
        }
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let gx = gy.matmul(&x[1].clone().transpose());
        let gw = x[0].clone().transpose().matmul(&gy);

        let gx = if gx.shape() != x[0].shape() {
            gx.sum_to(x[0].shape())
        } else {
            gx
        };
        let gw = if gw.shape() != x[1].shape() {
            gw.sum_to(x[1].shape())
        } else {
            gw
        };

        if self.bias {
            vec![gx, gw, gy.sum_to(x[2].shape())]
        } else {
            vec![gx, gw]
        }
    }
}

define!(Sigmoid);

impl Function for Sigmoid {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        (&x[0] * 0.5).tanh() * 0.5 + 0.5
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let y = self.output.as_ref().unwrap().get_array();
        vec![gy * &y * (1. - y)]
    }
}

define!(ReLU);

impl Function for ReLU {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].relu_max(0.)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
        vec![x.relu_mask(&gy, 0.)]
    }
}

define!(MeanSquaredError);

impl Function for MeanSquaredError {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        let diff = &x[0] - &x[1];
        diff.powi(2).sum() / diff.size() as f32
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let diff = &x[0] - &x[1];
        let gx = gy * &diff * (2. / diff.size() as f32);
        vec![gx.clone(), -gx]
    }
}

define!(Softmax, axis: usize);

fn softmax(x: &Array, axis: usize) -> Array {
    let y = (x - x.max(axis)).exp();
    &y / y.sum_with_axis(axis)
}

impl Function for Softmax {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        softmax(&x[0], self.axis)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let y = self.output.as_ref().unwrap().get_array();
        let gx = &y * gy;
        let sumdx = &gx.sum_with_axis(self.axis);
        vec![gx - y * sumdx]
    }
}

define!(CrossEntropy);

impl Function for CrossEntropy {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        -x[0].clip(1e-15, 1.).ln().matmul(&x[1].transpose()).sum()
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let cliped_x = &x[0].clip(1e-15, 1.);
        let gx = -&x[1] / cliped_x;
        let gt = -cliped_x.ln();
        vec![gx * &gy, gt * &gy]
    }
}

define!(SoftmaxCrossEntropy);

fn logsumexp(x: &Array, axis: usize) -> Array {
    let max_x = x.max(1);
    (x - &max_x).exp().sum_with_axis(axis).ln() + &max_x
}

impl Function for SoftmaxCrossEntropy {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        let (x, t) = (&x[0], &x[1]);
        let shape = x.shape();
        let n = shape[0];

        let log_z = logsumexp(x, 1);
        let log_p = x - &log_z;

        let mut loss = 0.0;
        for (row, &label) in t.get_data().iter().enumerate() {
            loss -= log_p[[row, label as usize]];
        }
        loss /= n as f32;

        Array::new(vec![loss], vec![])
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();

        let (x, t) = (&x[0], &x[1]);
        let n = x.shape()[0];

        let mut y = softmax(x, 1);
        for (row, &label) in t.get_data().iter().enumerate() {
            y[[row, label as usize]] -= 1.0;
        }
        y = y * gy / n as f32;

        vec![y]
    }
}
