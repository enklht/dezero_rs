use std::collections::HashMap;

use crate::{array::Array, layers::Model, variable::VBox};

pub trait Optimizer {
    fn update(&mut self) {
        let params = self.get_params();

        for param in params {
            self.update_one(param)
        }
    }
    fn get_params(&mut self) -> Vec<VBox>;
    fn update_one(&mut self, param: VBox);
}

pub struct SGD {
    lr: f32,
    target: Model,
}

impl SGD {
    pub fn new(lr: f32, target: Model) -> Self {
        SGD { lr, target }
    }
}

impl Optimizer for SGD {
    fn get_params(&mut self) -> Vec<VBox> {
        self.target.get_params()
    }
    fn update_one(&mut self, param: VBox) {
        param.set_array(param.get_array() - self.lr * param.get_grad())
    }
}

pub struct Momentum {
    lr: f32,
    momentum: f32,
    vs: HashMap<VBox, Array>,
    target: Model,
}

impl Momentum {
    pub fn new(lr: f32, momentum: f32, target: Model) -> Self {
        Momentum {
            lr,
            momentum,
            vs: HashMap::new(),
            target,
        }
    }
}

impl Optimizer for Momentum {
    fn get_params(&mut self) -> Vec<VBox> {
        self.target.get_params()
    }
    fn update_one(&mut self, param: VBox) {
        let v = self
            .vs
            .entry(param.clone())
            .or_insert_with(|| Array::zeros(&param.get_shape()));
        *v = &*v * self.momentum;
        *v = &*v - self.lr * param.get_grad();
        param.set_array(param.get_array() + &*v)
    }
}

pub struct Adam {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    t: f32,
    ms: HashMap<VBox, Array>,
    vs: HashMap<VBox, Array>,
    target: Model,
}

impl Adam {
    pub fn new(alpha: f32, beta1: f32, beta2: f32, target: Model) -> Self {
        Adam {
            alpha,
            beta1,
            beta2,
            t: 0.,
            ms: HashMap::new(),
            vs: HashMap::new(),
            target,
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self) {
        let params = self.get_params();

        self.t += 1.;
        for param in params {
            self.update_one(param)
        }
    }
    fn get_params(&mut self) -> Vec<VBox> {
        self.target.get_params()
    }
    fn update_one(&mut self, param: VBox) {
        let grad = param.get_grad();
        let m = self
            .ms
            .entry(param.clone())
            .or_insert_with(|| Array::zeros(&param.get_shape()));
        let v = self
            .vs
            .entry(param.clone())
            .or_insert_with(|| Array::zeros(&param.get_shape()));

        *m = self.beta1 * &*m + (1. - self.beta1) * &grad;
        *v = self.beta2 * &*v + (1. - self.beta2) * &grad * &grad;
        let m_hat = &*m / (1. - self.beta1.powf(self.t));
        let v_hat = &*v / (1. - self.beta1.powf(self.t));

        param.set_array(param.get_array() - self.alpha * m_hat / (v_hat.sqrt() + 1e-8))
    }
}
