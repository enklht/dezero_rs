use crate::array::Array;
use crate::functions::macros::*;
use crate::functions::trait_::Function;
use crate::variable::{VBox, WeakVBox};

define!(Neg,);

impl Function for Neg {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        -x[0].clone()
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        vec![-gy]
    }
}

define!(Powi, n: i32);

impl Function for Powi {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].powi(self.n)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
        vec![self.n as f32 * x.powi(self.n - 1) * gy]
    }
}

define!(Powf, c: f32);

impl Function for Powf {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].powf(self.c)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
        vec![self.c * x.powf(self.c - 1.) * gy]
    }
}

define!(Exp,);

impl Function for Exp {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].exp()
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x = self.inputs.as_ref().unwrap().first().unwrap().get_array();
        vec![x.exp() * gy]
    }
}

define!(Reshape, shape_in: Vec<usize>, shape_out: Vec<usize>);

impl Function for Reshape {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].clone().reshape(&self.shape_out)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        vec![gy.reshape(&self.shape_in)]
    }
}

define!(Transpose,);

impl Function for Transpose {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].clone().transpose()
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        vec![gy.transpose()]
    }
}
