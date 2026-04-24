use std::{cell::RefCell, rc::Rc};

use crate::functions as F;
use crate::{
    array::Array,
    variable::{VBox, WeakVBox},
};

#[derive(Clone)]
pub struct Model(Rc<RefCell<dyn Layer>>);

impl Model {
    pub fn new(target: impl Layer + 'static) -> Model {
        Model(Rc::new(RefCell::new(target)))
    }

    pub fn clear_grads(&self) {
        self.0.borrow_mut().clear_grads();
    }

    pub fn call(&self, x: &VBox) -> VBox {
        self.0.borrow_mut().call(x)
    }

    pub fn get_params(&self) -> Vec<VBox> {
        self.0.borrow().get_params()
    }
}

pub trait Layer {
    fn call(&mut self, x: &VBox) -> VBox {
        let y = self.forward(x);
        self.set_io(x, &y);
        y
    }
    fn forward(&mut self, x: &VBox) -> VBox;
    fn clear_grads(&mut self);
    fn set_io(&mut self, input: &VBox, output: &VBox);
    fn get_params(&self) -> Vec<VBox>;
}

pub struct MLP {
    input: Option<WeakVBox>,
    output: Option<WeakVBox>,
    out_sizes: Vec<usize>,
    activation: Box<dyn Fn(&VBox) -> VBox>,
    layers: Vec<Linear>,
}

impl MLP {
    pub fn new(out_sizes: &[usize], activation: Box<dyn Fn(&VBox) -> VBox>) -> MLP {
        let mut layers = Vec::new();
        for out_size in out_sizes {
            layers.push(Linear::new(*out_size, true))
        }
        MLP {
            input: None,
            output: None,
            out_sizes: out_sizes.to_vec(),
            activation,
            layers,
        }
    }

    pub fn get_out_sizes(&self) -> Vec<usize> {
        self.out_sizes.to_vec()
    }
}

impl Layer for MLP {
    fn forward(&mut self, x: &VBox) -> VBox {
        let mut y = x.clone();
        let len = self.layers.len();
        for layer in &mut self.layers[..len - 1] {
            y = (self.activation)(&layer.call(&y))
        }
        self.layers[len - 1].call(&y)
    }
    fn set_io(&mut self, input: &VBox, output: &VBox) {
        self.input = Some(input.clone().downgrade());
        self.output = Some(output.clone().downgrade())
    }
    fn get_params(&self) -> Vec<VBox> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.append(&mut layer.get_params());
        }
        params
    }
    fn clear_grads(&mut self) {
        for layer in &mut self.layers {
            layer.clear_grads();
        }
    }
}

pub struct Linear {
    inputs: Option<WeakVBox>,
    outputs: Option<WeakVBox>,
    out_size: usize,
    w: Option<VBox>,
    b: Option<VBox>,
}

impl Linear {
    pub fn new(out_size: usize, bias: bool) -> Self {
        let b = if bias {
            Some(VBox::new(Array::zeros(&[out_size])))
        } else {
            None
        };

        Linear {
            inputs: None,
            outputs: None,
            out_size,
            w: None,
            b,
        }
    }

    fn init_w(&mut self, in_size: usize) {
        let w = VBox::new(Array::randn(
            &[in_size, self.out_size],
            0.,
            (2.0 / in_size as f32).sqrt(),
        ));
        self.w = Some(w);
    }
}

impl Layer for Linear {
    fn forward(&mut self, x: &VBox) -> VBox {
        if self.w.is_none() {
            self.init_w(x.get_shape()[1]);
        }
        F::linear(x, self.w.as_ref().unwrap(), self.b.as_ref())
    }
    fn clear_grads(&mut self) {
        self.w.as_ref().unwrap().clear_grad();
        if let Some(b) = &self.b {
            b.clear_grad();
        }
    }
    fn set_io(&mut self, input: &VBox, output: &VBox) {
        self.inputs = Some(input.clone().downgrade());
        self.outputs = Some(output.clone().downgrade());
    }
    fn get_params(&self) -> Vec<VBox> {
        let mut params = Vec::new();
        if let Some(w) = self.w.as_ref() {
            params.push(w.clone());
        }
        if let Some(b) = self.b.as_ref() {
            params.push(b.clone())
        }
        params
    }
}
