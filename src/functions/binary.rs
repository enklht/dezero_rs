use crate::array::Array;
use crate::functions::macros::*;
use crate::functions::trait_::Function;
use crate::variable::{VBox, WeakVBox};

pub struct Add {
    inputs: Option<Vec<VBox>>,
    output: Option<WeakVBox>,
    generation: u32,
    shape0: Vec<usize>,
    shape1: Vec<usize>,
}

impl Add {
    pub fn new(shape0: Vec<usize>, shape1: Vec<usize>) -> Self {
        Self {
            inputs: None,
            output: None,
            generation: 0,
            shape0,
            shape1,
        }
    }
}

impl Function for Add {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        &x[0] + &x[1]
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let gx0 = gy.clone();
        let gx1 = gy;
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

pub struct Sub {
    inputs: Option<Vec<VBox>>,
    output: Option<WeakVBox>,
    generation: u32,
    shape0: Vec<usize>,
    shape1: Vec<usize>,
}

impl Sub {
    pub fn new(shape0: Vec<usize>, shape1: Vec<usize>) -> Self {
        Self {
            inputs: None,
            output: None,
            generation: 0,
            shape0,
            shape1,
        }
    }
}

impl Function for Sub {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        &x[0] - &x[1]
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let gx0 = gy.clone();
        let gx1 = -gy;
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

pub struct Mul {
    inputs: Option<Vec<VBox>>,
    output: Option<WeakVBox>,
    generation: u32,
    shape0: Vec<usize>,
    shape1: Vec<usize>,
}

impl Mul {
    pub fn new(shape0: Vec<usize>, shape1: Vec<usize>) -> Self {
        Self {
            inputs: None,
            output: None,
            generation: 0,
            shape0,
            shape1,
        }
    }
}

impl Function for Mul {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        &x[0] * &x[1]
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let gx0 = &x[1] * &gy;
        let gx1 = &x[0] * gy;
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}

pub struct Div {
    inputs: Option<Vec<VBox>>,
    output: Option<WeakVBox>,
    generation: u32,
    shape0: Vec<usize>,
    shape1: Vec<usize>,
}

impl Div {
    pub fn new(shape0: Vec<usize>, shape1: Vec<usize>) -> Self {
        Self {
            inputs: None,
            output: None,
            generation: 0,
            shape0,
            shape1,
        }
    }
}

impl Function for Div {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        &x[0] / &x[1]
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        let gx0 = &gy / &x[1];
        let gx1 = -&x[0] / (&x[1] * &x[1]) * gy;
        if self.shape0 != self.shape1 {
            vec![gx0.sum_to(&self.shape0), gx1.sum_to(&self.shape1)]
        } else {
            vec![gx0, gx1]
        }
    }
}
