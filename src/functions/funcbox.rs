use crate::array::Array;
use crate::functions::trait_::Function;
use crate::variable::{VBox, WeakVBox};
use std::hash::Hash;
use std::rc::Rc;

#[derive(Clone)]
pub struct FuncBox(pub Rc<dyn Function>);

impl FuncBox {
    pub fn get_gen(&self) -> u32 {
        self.0.get_generation()
    }

    pub fn get_inputs(&self) -> Vec<VBox> {
        self.0.get_inputs()
    }

    pub fn get_output(&self) -> WeakVBox {
        self.0.get_output()
    }

    pub fn backward(&self, gy: Array) -> Vec<Array> {
        self.0.backward(gy)
    }
}

impl PartialEq for FuncBox {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for FuncBox {}

impl Hash for FuncBox {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize((Rc::as_ptr(&self.0) as *mut usize) as usize);
    }
}

impl Ord for FuncBox {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.get_generation().cmp(&other.0.get_generation())
    }
}

impl PartialOrd for FuncBox {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
