use crate::array::Array;
use crate::functions::macros::*;
use crate::functions::trait_::Function;
use crate::variable::{VBox, WeakVBox};

define!(Sum, shape: Vec<usize>);

impl Function for Sum {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].sum()
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        vec![gy.broadcast_to(&self.shape)]
    }
}

define!(SumTo, shape_in: Vec<usize>, shape_out: Vec<usize>);

impl Function for SumTo {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].clone().sum_to(&self.shape_out)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        vec![gy.broadcast_to(&self.shape_in)]
    }
}

define!(BroadcastTo, shape_in: Vec<usize>, shape_out: Vec<usize>);

impl Function for BroadcastTo {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].clone().broadcast_to(&self.shape_out)
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        vec![gy.sum_to(&self.shape_in)]
    }
}

define!(Matmul,);

impl Function for Matmul {
    impl_getters_setters!();
    fn forward(&self, x: Vec<Array>) -> Array {
        x[0].matmul(&x[1])
    }
    fn backward(&self, gy: Array) -> Vec<Array> {
        let x: Vec<Array> = self
            .inputs
            .as_ref()
            .unwrap()
            .iter()
            .map(|x| x.get_array())
            .collect();
        vec![
            gy.matmul(&x[1].clone().transpose()),
            x[0].clone().transpose().matmul(&gy),
        ]
    }
}
