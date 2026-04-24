use crate::functions as F;
use crate::functions::FuncBox;
use std::ops::{Add, Div, Mul, Neg, Sub};
use std::{
    cell::RefCell,
    collections::{BinaryHeap, HashSet},
    hash::Hash,
    rc::{Rc, Weak},
};

use crate::array::Array;

#[macro_export]
macro_rules! scaler {
    ($x: expr) => {
        &$crate::variable::VBox::new($crate::array::Array::new(vec![$x as f32], vec![]))
    };
}

#[macro_export]
macro_rules! var {
    ($x: expr) => {
        &$crate::variable::VBox::new($x)
    };
}

pub struct Variable {
    array: Array,
    grad: Option<Array>,
    creator: Option<FuncBox>,
    generation: u32,
}

#[derive(Clone)]
pub struct VBox(Rc<RefCell<Variable>>);

#[derive(Clone)]
pub struct WeakVBox(Weak<RefCell<Variable>>);

impl PartialEq for VBox {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

impl Eq for VBox {}

impl Hash for VBox {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_usize((Rc::as_ptr(&self.0) as *mut usize) as usize);
    }
}

impl WeakVBox {
    pub fn new(weak: Weak<RefCell<Variable>>) -> WeakVBox {
        WeakVBox(weak)
    }

    pub fn upgrade(&self) -> VBox {
        VBox::from_rc(self.0.upgrade().unwrap())
    }

    pub fn get_array(&self) -> Array {
        let v = self.upgrade();
        v.get_array()
    }

    pub fn get_grad(&self) -> Array {
        let v = self.upgrade();
        v.get_grad()
    }

    pub fn clear_grad(&self) {
        let v = self.upgrade();
        v.clear_grad();
    }
}

impl VBox {
    pub fn new(array: Array) -> VBox {
        VBox(Rc::new(RefCell::new(Variable {
            array,
            grad: None,
            creator: None,
            generation: 0,
        })))
    }

    pub fn from_rc(rc: Rc<RefCell<Variable>>) -> VBox {
        VBox(rc)
    }

    pub fn get_array(&self) -> Array {
        let v = self.0.as_ref();
        v.borrow().array.clone()
    }

    pub fn get_shape(&self) -> Vec<usize> {
        let v = self.0.as_ref();
        v.borrow().array.get_shape().to_vec()
    }

    pub fn get_grad(&self) -> Array {
        let v = self.0.as_ref();
        v.borrow().grad.clone().unwrap()
    }

    pub fn get_option_grad(&self) -> Option<Array> {
        let v = self.0.as_ref();
        v.borrow().grad.clone()
    }

    pub fn get_creator(&self) -> Option<FuncBox> {
        self.0.clone().borrow().creator.clone()
    }

    pub fn get_gen(&self) -> u32 {
        self.0.clone().borrow().generation
    }

    pub fn set_array(&self, data: Array) {
        let v = self.0.as_ref();
        v.borrow_mut().array.set_data(data);
    }

    pub fn set_grad(&self, grad: Array) {
        let v = self.0.as_ref();
        match &mut v.borrow_mut().grad {
            Some(grad_old) => grad_old.set_data(grad),
            x @ None => *x = Some(grad),
        };
    }

    pub fn clear_grad(&self) {
        let v = self.0.as_ref();
        v.borrow_mut().grad = None;
    }

    pub fn set_creator(&self, func: FuncBox) {
        let tmp = self.0.as_ref();
        let mut v = tmp.borrow_mut();
        v.generation = func.get_gen() + 1;
        v.creator = Some(func);
    }

    pub fn backward(&self) {
        self.backward_with_option(false);
    }

    pub fn backward_with_option(&self, retain_grad: bool) {
        if self.get_option_grad().is_none() {
            self.set_grad(Array::ones(&self.get_shape()));
        }

        let mut funcs = BinaryHeap::new();
        let mut seen_set = HashSet::new();

        let creator = self
            .get_creator()
            .expect("The creator of the variable is not set.\nMaybe backpropagation is disabled.");
        funcs.push(creator.clone());
        seen_set.insert(creator);

        while let Some(f) = funcs.pop() {
            let x = f.get_inputs();
            let gy = f.get_output().get_grad();
            let gxs = f.backward(gy);

            for (x, gx) in x.iter().zip(gxs.into_iter()) {
                if let Some(gx_old) = x.get_option_grad() {
                    x.set_grad(gx_old + &gx)
                } else {
                    x.set_grad(gx);
                }

                if let Some(x_creator) = x.get_creator() {
                    if !seen_set.contains(&x_creator) {
                        funcs.push(x_creator.clone());
                        seen_set.insert(x_creator);
                    }
                }
            }

            if !retain_grad {
                f.get_output().clear_grad();
            }
        }
    }

    pub fn downgrade(self) -> WeakVBox {
        WeakVBox::new(Rc::downgrade(&self.0))
    }

    pub fn powi(&self, n: i32) -> VBox {
        let func = F::Powi::new(n);
        F::call(func, std::slice::from_ref(self))
    }

    pub fn pow(&self, c: f32) -> VBox {
        let func = F::Powf::new(c);
        F::call(func, std::slice::from_ref(self))
    }

    pub fn exp(&self) -> VBox {
        let func = F::Exp::new();
        F::call(func, std::slice::from_ref(self))
    }

    pub fn reshape(&self, shape: Vec<usize>) -> VBox {
        let func = F::Reshape::new(self.get_shape(), shape);
        F::call(func, std::slice::from_ref(self))
    }

    pub fn transpose(&self) -> VBox {
        let func = F::Transpose::new();
        F::call(func, std::slice::from_ref(self))
    }

    pub fn sum(&self) -> VBox {
        let func = F::Sum::new(self.get_shape());
        F::call(func, std::slice::from_ref(self))
    }

    pub fn sum_to(&self, shape: &[usize]) -> VBox {
        let func = F::SumTo::new(self.get_shape(), shape.to_vec());
        F::call(func, std::slice::from_ref(self))
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> VBox {
        let func = F::BroadcastTo::new(self.get_shape(), shape.to_vec());
        F::call(func, std::slice::from_ref(self))
    }

    pub fn matmul(&self, rhs: &VBox) -> VBox {
        let func = F::Matmul::new();
        F::call(func, &[self.clone(), rhs.clone()])
    }
}

impl std::fmt::Display for VBox {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut string = format!("Variable({}", self.get_array().to_string(9));
        match self.get_option_grad() {
            None => {}
            Some(g) => string += &format!(",\n   grad: {}", g.to_string(9)),
        }
        string += ")";
        write!(f, "{}", string)
    }
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                let func = F::$trait::new(self.get_shape(), rhs.get_shape());
                F::call(func, &[self, rhs]).clone()
            }
        }

        impl $trait<VBox> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                VBox::$fname(self.clone(), rhs)
            }
        }

        impl $trait<&VBox> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                VBox::$fname(self, rhs.clone())
            }
        }

        impl $trait for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: Self) -> Self::Output {
                VBox::$fname(self.clone(), rhs.clone())
            }
        }

        impl $trait<f32> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: f32) -> Self::Output {
                self.$fname(scaler!(rhs))
            }
        }

        impl $trait<f32> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: f32) -> Self::Output {
                self.$fname(crate::scaler!(rhs))
            }
        }

        impl $trait<VBox> for f32 {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }

        impl $trait<&VBox> for f32 {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }

        impl $trait<i32> for VBox {
            type Output = VBox;
            fn $fname(self, rhs: i32) -> Self::Output {
                self.$fname(crate::scaler!(rhs))
            }
        }

        impl $trait<i32> for &VBox {
            type Output = VBox;
            fn $fname(self, rhs: i32) -> Self::Output {
                self.$fname(crate::scaler!(rhs))
            }
        }

        impl $trait<VBox> for i32 {
            type Output = VBox;
            fn $fname(self, rhs: VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }

        impl $trait<&VBox> for i32 {
            type Output = VBox;
            fn $fname(self, rhs: &VBox) -> Self::Output {
                crate::scaler!(self).$fname(rhs)
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Mul, mul);
impl_op!(Sub, sub);
impl_op!(Div, div);

impl Neg for VBox {
    type Output = VBox;
    fn neg(self) -> Self::Output {
        let func = F::Neg::new();
        F::call(func, &[self])
    }
}

impl Neg for &VBox {
    type Output = VBox;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}
