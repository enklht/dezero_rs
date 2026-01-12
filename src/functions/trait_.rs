use crate::{
    array::Array,
    variable::{VBox, WeakVBox},
};

pub trait Function {
    fn get_generation(&self) -> u32;
    fn get_inputs(&self) -> Vec<VBox>;
    fn get_output(&self) -> WeakVBox;

    fn set_generation(&mut self, gen: u32);
    fn set_inputs(&mut self, inputs: Vec<VBox>);
    fn set_output(&mut self, output: WeakVBox);

    fn forward(&self, x: Vec<Array>) -> Array;
    fn backward(&self, gy: Array) -> Vec<Array>;
}
