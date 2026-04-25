use super::{utils::*, Array};
use ndarray::{ArrayD, ArrayView1, ArrayView2, Axis, Ix1, Ix2};
use std::ops::{Add, Div, Mul, Neg, Sub};

macro_rules! define_map_functions {
    ($($fname: ident),*) => {
        $(
        pub fn $fname(&self) -> Array {
            self.map(f32::$fname)
        })*
    };
}

impl Array {
    pub fn map<F>(&self, f: F) -> Array
    where
        F: Fn(f32) -> f32,
    {
        let data = self.data.mapv(f);
        Array { data }
    }

    define_map_functions!(exp, ln, sin, cos, tan, sinh, cosh, tanh);

    pub fn powi(&self, n: i32) -> Array {
        let data = self.data.mapv(|a| a.powi(n));
        Array { data }
    }

    pub fn powf(&self, n: f32) -> Array {
        let data = self.data.mapv(|a| a.powf(n));
        Array { data }
    }

    pub fn sum(&self) -> Array {
        let data = ArrayD::from_elem(vec![], self.data.sum());
        Array { data }
    }

    pub fn sum_to(&self, shape: &[usize]) -> Array {
        if self.data.shape() == shape {
            return self.clone();
        }

        let Some(lead) = self.data.ndim().checked_sub(shape.len()) else {
            panic!("failed to sum {:?} to {:?}", shape, self.data.shape())
        };

        let mut new_shape = vec![1; lead];
        new_shape.extend_from_slice(shape);

        let mut axes = Vec::new();
        for (axis, (i, j)) in self.data.shape().iter().zip(new_shape.iter()).enumerate() {
            match (i, j) {
                (i, j) if i == j => {}
                (_, 1) => axes.push(axis),
                _ => panic!("failed to sum {:?} to {:?}", self.data.shape(), shape),
            }
        }

        let mut data = self.data.clone();
        for axis in axes.into_iter().rev() {
            data = data.sum_axis(Axis(axis)).insert_axis(Axis(axis));
        }

        let data = data
            .into_shape_with_order(shape.to_vec())
            .expect("Invalid shape");
        Array { data }
    }

    pub fn sum_with_axis(&self, axis: usize) -> Array {
        let mut shape = self.data.shape().to_vec();
        shape[axis] = 1;
        self.sum_to(&shape)
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Array {
        if self.data.shape() == shape {
            return self.clone();
        }
        let Some(view) = self.data.broadcast(shape.to_vec()) else {
            panic!("failed to broadcast {:?} to {:?}", self.data.shape(), shape)
        };
        Array {
            data: view.to_owned(),
        }
    }

    pub fn reshape(&self, new_shape: &[usize]) -> Array {
        let new_size: usize = new_shape.iter().product();
        if self.data.len() != new_size {
            panic!("Cannot convert {:?} to {:?}", self.data.shape(), new_shape)
        }
        let data = self
            .data
            .clone()
            .into_shape_with_order(new_shape.to_vec())
            .expect("Invalid shape");
        Array { data }
    }

    pub fn transpose(&self) -> Array {
        let dim = self.data.ndim();
        if dim < 2 {
            return self.clone();
        }
        let mut axes: Vec<usize> = (0..dim).collect();
        axes.swap(dim - 2, dim - 1);
        Array {
            data: self.data.clone().permuted_axes(axes),
        }
    }

    pub fn matmul(&self, rhs: &Array) -> Array {
        let ldim = self.data.ndim();
        let rdim = rhs.data.ndim();
        if ldim == 0 || rdim == 0 {
            panic!("Scaler cannot be multiplied with matrices using matmul.")
        }

        if ldim == 1 && rdim == 1 {
            let val = as_ix1(&self.data).dot(&as_ix1(&rhs.data));
            let data = ArrayD::from_elem(vec![], val);
            return Array { data };
        }
        if ldim == 1 && rdim == 2 {
            let out = as_ix1(&self.data).dot(&as_ix2(&rhs.data));
            let shape = out.shape().to_vec();
            let data = out.iter().cloned().collect::<Vec<_>>();
            return Array::new(data, shape);
        }
        if ldim == 2 && rdim == 1 {
            let out = as_ix2(&self.data).dot(&as_ix1(&rhs.data));
            let shape = out.shape().to_vec();
            let data = out.iter().cloned().collect::<Vec<_>>();
            return Array::new(data, shape);
        }
        if ldim == 2 && rdim == 2 {
            let out = as_ix2(&self.data).dot(&as_ix2(&rhs.data));
            let shape = out.shape().to_vec();
            let data = out.iter().cloned().collect::<Vec<_>>();
            return Array::new(data, shape);
        }

        matmul_general(self, rhs)
    }

    pub fn relu_max(&self, rhs: f32) -> Array {
        let data = self.data.mapv(|a| a.max(rhs));
        Array { data }
    }

    pub fn relu_mask(&self, rhs: &Array, threshold: f32) -> Array {
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&x, &y)| if x > threshold { y } else { 0. })
            .collect::<Vec<_>>();
        Array::new(data, self.data.shape().to_vec())
    }

    pub fn clip(&self, lowerbound: f32, upperbound: f32) -> Array {
        let data = self.data.mapv(|x| x.min(upperbound).max(lowerbound));
        Array { data }
    }

    pub fn max(&self, axis: usize) -> Array {
        let data = self
            .data
            .map_axis(Axis(axis), |v| v.fold(f32::NEG_INFINITY, |x, y| x.max(*y)))
            .insert_axis(Axis(axis));
        Array { data }
    }
}

fn matmul_general(lhs: &Array, rhs: &Array) -> Array {
    let ldim = lhs.data.ndim();
    let rdim = rhs.data.ndim();

    let (l_squeeze_flag, r_squeeze_flag): (bool, bool);
    let (l, m, m_, n): (usize, usize, usize, usize);
    let (lstackshape, rstackshape): (Vec<usize>, Vec<usize>);

    if ldim == 1 {
        l_squeeze_flag = true;
        l = 1;
        m = lhs.data.shape()[0];
        lstackshape = Vec::new();
    } else {
        l_squeeze_flag = false;
        l = lhs.data.shape()[ldim - 2];
        m = lhs.data.shape()[ldim - 1];
        lstackshape = lhs.data.shape()[..ldim - 2].to_vec();
    };

    if rdim == 1 {
        r_squeeze_flag = true;
        m_ = rhs.data.shape()[0];
        n = 1;
        rstackshape = Vec::new();
    } else {
        r_squeeze_flag = false;
        m_ = rhs.data.shape()[rdim - 2];
        n = rhs.data.shape()[rdim - 1];
        rstackshape = rhs.data.shape()[..rdim - 2].to_vec();
    };

    if m != m_ {
        panic!("invalid shape")
    }

    let stack_shape = shape_after_broadcast(&lstackshape, &rstackshape).unwrap();
    let mut new_shape = stack_shape.clone();
    new_shape.extend([l, n]);

    let lhs_target = stack_shape
        .iter()
        .cloned()
        .chain([l, m])
        .collect::<Vec<_>>();
    let rhs_target = stack_shape
        .iter()
        .cloned()
        .chain([m, n])
        .collect::<Vec<_>>();

    let mut lhs_view = lhs.data.view();
    if ldim == 1 {
        lhs_view = lhs_view.insert_axis(Axis(0));
    }
    while lhs_view.ndim() < lhs_target.len() {
        lhs_view = lhs_view.insert_axis(Axis(0));
    }

    let mut rhs_view = rhs.data.view();
    if rdim == 1 {
        rhs_view = rhs_view.insert_axis(Axis(1));
    }
    while rhs_view.ndim() < rhs_target.len() {
        rhs_view = rhs_view.insert_axis(Axis(0));
    }

    let lhs = lhs_view
        .broadcast(lhs_target.clone())
        .expect("invalid broadcast for lhs")
        .to_owned();
    let rhs = rhs_view
        .broadcast(rhs_target.clone())
        .expect("invalid broadcast for rhs")
        .to_owned();

    let num_batches: usize = stack_shape.iter().product();
    let lhs = lhs
        .to_shape((num_batches, l, m))
        .expect("invalid shape")
        .to_owned();
    let rhs = rhs
        .to_shape((num_batches, m, n))
        .expect("invalid shape")
        .to_owned();

    let mut out = Vec::with_capacity(num_batches * l * n);
    for b in 0..num_batches {
        let lhs_b = lhs.index_axis(Axis(0), b);
        let rhs_b = rhs.index_axis(Axis(0), b);
        let out_b = lhs_b.dot(&rhs_b);
        out.extend(out_b.iter().cloned());
    }

    if l_squeeze_flag && r_squeeze_flag {
        new_shape.pop();
        new_shape.pop();
    } else if l_squeeze_flag {
        new_shape.remove(new_shape.len() - 2);
    } else if r_squeeze_flag {
        new_shape.pop();
    }

    Array::new(out, new_shape)
}

fn as_ix1(data: &ArrayD<f32>) -> ArrayView1<'_, f32> {
    data.view()
        .into_dimensionality::<Ix1>()
        .expect("invalid shape")
}

fn as_ix2(data: &ArrayD<f32>) -> ArrayView2<'_, f32> {
    data.view()
        .into_dimensionality::<Ix2>()
        .expect("invalid shape")
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for &Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
                if self.data.ndim() == 0 {
                    return self.data[[]].$fname(rhs);
                }
                if rhs.data.ndim() == 0 {
                    return self.$fname(rhs.data[[]]);
                }
                if self.data.shape() != rhs.data.shape() {
                    let new_shape = shape_after_broadcast(self.data.shape(), rhs.data.shape())
                        .expect(&format!(
                            "Two arrays must have the same shape\nlhs: {:?}\nrhs: {:?}",
                            self, rhs,
                        ));

                    let ldata = self.broadcast_to(&new_shape).data;
                    let rdata = rhs.broadcast_to(&new_shape).data;

                    let data = ldata
                        .iter()
                        .zip(rdata.iter())
                        .map(|(x, y)| f32::$fname(*x, *y))
                        .collect::<Vec<_>>();
                    Array::new(data, new_shape)
                } else {
                    let data = self
                        .data
                        .iter()
                        .zip(rhs.data.iter())
                        .map(|(x, y)| f32::$fname(*x, *y))
                        .collect::<Vec<_>>();
                    Array::new(data, self.data.shape().to_vec())
                }
            }
        }

        impl $trait<Array> for &Array {
            type Output = Array;
            fn $fname(self, rhs: Array) -> Self::Output {
                self.$fname(&rhs)
            }
        }

        impl $trait<&Array> for Array {
            type Output = Array;
            fn $fname(self, rhs: &Array) -> Self::Output {
                (&self).$fname(rhs)
            }
        }

        impl $trait for Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
                self.$fname(&rhs)
            }
        }

        impl $trait<f32> for Array {
            type Output = Array;
            fn $fname(self, rhs: f32) -> Self::Output {
                (&self).$fname(rhs)
            }
        }

        impl $trait<Array> for f32 {
            type Output = Array;
            fn $fname(self, rhs: Array) -> Self::Output {
                self.$fname(&rhs)
            }
        }

        impl $trait<f32> for &Array {
            type Output = Array;
            fn $fname(self, rhs: f32) -> Self::Output {
                let data = self.data.mapv(|x| f32::$fname(x, rhs));
                Array { data }
            }
        }

        impl $trait<&Array> for f32 {
            type Output = Array;
            fn $fname(self, rhs: &Array) -> Self::Output {
                let data = rhs.data.mapv(|x| f32::$fname(self, x));
                Array { data }
            }
        }
    };
}

impl_op!(Add, add);
impl_op!(Sub, sub);
impl_op!(Mul, mul);
impl_op!(Div, div);

impl Neg for Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        let data = self.data.mapv(|a| -a);
        Array { data }
    }
}

impl Neg for &Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}
