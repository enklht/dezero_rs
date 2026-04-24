use super::{utils::*, Array};
use std::ops::{Add, Div, Mul, Neg, Sub};

fn unravel_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coords
}

fn batch_index_from_broadcast(coords: &[usize], src_shape: &[usize], lead: usize) -> usize {
    if src_shape.is_empty() {
        return 0;
    }

    let mut index = 0;
    for (axis, &dim) in src_shape.iter().enumerate() {
        let coord = if dim == 1 { 0 } else { coords[lead + axis] };
        index = index * dim + coord;
    }
    index
}

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
        let data = self.data.iter().map(|&a| f(a)).collect();
        Array::new(data, self.shape.clone())
    }

    define_map_functions!(exp, ln, sin, cos, tan, sinh, cosh, tanh);

    pub fn powi(&self, n: i32) -> Array {
        let data = self.data.iter().map(|a| a.powi(n)).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn powf(&self, n: f32) -> Array {
        let data = self.data.iter().map(|a| a.powf(n)).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn sum(&self) -> Array {
        Array {
            data: vec![self.data.iter().sum()],
            shape: vec![],
            size: 1,
        }
    }

    pub fn sum_to(&self, shape: &[usize]) -> Array {
        if self.shape == shape {
            return self.clone();
        }

        let Some(lead) = self.shape.len().checked_sub(shape.len()) else {
            panic!("failed to sum {:?} to {:?}", shape, self.shape)
        };

        let tmp = vec![1; lead];
        let new_shape = tmp
            .into_iter()
            .chain(shape.iter().cloned())
            .collect::<Vec<_>>();

        let mut axes = Vec::new();
        for (axis, (i, j)) in self.shape.iter().zip(new_shape.iter()).enumerate() {
            match (i, j) {
                (i, j) if i == j => {}
                (_, 1) => {
                    axes.push(axis);
                }
                _ => panic!("failed to sum {:?} to {:?}", self.shape, shape),
            }
        }

        let mut data = self.data.clone();
        let dim = self.shape.len();

        let mut steps = vec![1];
        for x in self.shape.iter().rev() {
            steps.push(steps.last().unwrap() * x);
        }

        for axis in axes {
            let mut new_data = Vec::new();
            let step = steps[dim - axis - 1];
            for i in 0..(data.len() / steps[dim - axis]) {
                for j in 0..step {
                    new_data.push(
                        data[steps[dim - axis] * i + j..]
                            .iter()
                            .step_by(step)
                            .take(self.shape[axis])
                            .sum(),
                    )
                }
            }
            data = new_data;
        }

        Array::new(data, shape.to_vec())
    }

    pub fn sum_with_axis(&self, axis: usize) -> Array {
        let mut shape = self.shape.clone();
        shape[axis] = 1;
        self.sum_to(&shape)
    }

    pub fn broadcast_to(&self, shape: &[usize]) -> Array {
        if self.shape == shape {
            return self.clone();
        }
        let Some(lead) = shape.len().checked_sub(self.shape.len()) else {
            panic!("failed to broadcast {:?} to {:?}", self.shape, shape)
        };

        let tmp = vec![1; lead];
        let old_shape = tmp
            .into_iter()
            .chain(self.shape.clone())
            .collect::<Vec<_>>();

        let data = broadcast_to(&self.data, &old_shape, shape);

        Array::new(data, shape.to_vec())
    }

    pub fn reshape(self, new_shape: &[usize]) -> Array {
        let new_size = new_shape.iter().product();
        if self.size != new_size {
            panic!("Cannot convert {:?} to {:?}", self.shape, new_shape)
        }
        Array {
            data: self.data,
            shape: new_shape.to_vec(),
            size: self.size,
        }
    }

    pub fn transpose(&self) -> Array {
        match self.shape.len() {
            0 | 1 => self.clone(),
            2 => self.transpose2d(),
            _ => self.transpose_nd(),
        }
    }

    fn transpose2d(&self) -> Array {
        let (m, n) = (self.shape[0], self.shape[1]);
        let mut data = Vec::with_capacity(self.size);
        for i in 0..n {
            for j in 0..m {
                data.push(self.data[n * j + i])
            }
        }
        Array {
            data,
            shape: vec![n, m],
            size: m * n,
        }
    }

    fn transpose_nd(&self) -> Array {
        let dim = self.shape.len();
        let m = self.shape[dim - 2];
        let n = self.shape[dim - 1];
        let batch = self.size / (m * n);

        let mut data = Vec::with_capacity(self.size);
        for b in 0..batch {
            let offset = b * m * n;
            for i in 0..n {
                for j in 0..m {
                    data.push(self.data[offset + n * j + i]);
                }
            }
        }

        let mut shape = self.shape.clone();
        shape[dim - 2] = n;
        shape[dim - 1] = m;

        Array {
            data,
            shape,
            size: self.size,
        }
    }

    pub fn matmul(&self, rhs: &Array) -> Array {
        let ldim = self.shape.len();
        let rdim = rhs.shape.len();
        if ldim == 0 || rdim == 0 {
            panic!("Scaler cannot be multiplied with matrices using matmul.")
        }

        let (l_squeeze_flag, r_squeeze_flag): (bool, bool);
        let (l, m, m_, n): (usize, usize, usize, usize);
        let (lstackshape, rstackshape): (&[usize], &[usize]);

        if ldim == 1 {
            l_squeeze_flag = true;
            l = 1;
            m = self.shape[0];
            lstackshape = &[];
        } else {
            l_squeeze_flag = false;
            l = self.shape[ldim - 2];
            m = self.shape[ldim - 1];
            lstackshape = &self.shape[..ldim - 2]
        };

        if rdim == 1 {
            r_squeeze_flag = true;
            m_ = rhs.shape[0];
            n = 1;
            rstackshape = &[];
        } else {
            r_squeeze_flag = false;
            m_ = rhs.shape[rdim - 2];
            n = rhs.shape[rdim - 1];
            rstackshape = &rhs.shape[..rdim - 2]
        };

        if m != m_ {
            panic!("invalid shape")
        }

        let stack_shape = shape_after_broadcast(lstackshape, rstackshape).unwrap();
        let mut new_shape = stack_shape.clone();
        new_shape.extend([l, n]);

        let new_size: usize = new_shape.iter().product();
        let num_batches: usize = stack_shape.iter().product();
        let out_stack_dim = stack_shape.len();
        let lhs_lead = out_stack_dim - lstackshape.len();
        let rhs_lead = out_stack_dim - rstackshape.len();

        let lhs_chunk_size = l * m;
        let rhs_chunk_size = m * n;

        let mut data = Vec::with_capacity(new_size);
        for batch in 0..num_batches {
            let coords = unravel_index(batch, &stack_shape);
            let lhs_batch = batch_index_from_broadcast(&coords, lstackshape, lhs_lead);
            let rhs_batch = batch_index_from_broadcast(&coords, rstackshape, rhs_lead);

            let lhs_start = lhs_batch * lhs_chunk_size;
            let rhs_start = rhs_batch * rhs_chunk_size;
            let lhs = &self.data[lhs_start..lhs_start + lhs_chunk_size];
            let rhs = &rhs.data[rhs_start..rhs_start + rhs_chunk_size];

            data.extend(matmul_2d(lhs, rhs, (l, m, n)));
        }

        if l_squeeze_flag && r_squeeze_flag {
            new_shape.pop();
            new_shape.pop();
        } else if l_squeeze_flag {
            new_shape.remove(new_shape.len() - 2);
        } else if r_squeeze_flag {
            new_shape.pop();
        }

        Array::new(data, new_shape)
    }

    pub fn relu_max(&self, rhs: f32) -> Array {
        let data = self.data.iter().map(|a| a.max(rhs)).collect();
        Array::new(data, self.shape.clone())
    }

    pub fn relu_mask(&self, rhs: &Array, threshold: f32) -> Array {
        let data = self
            .data
            .iter()
            .zip(rhs.data.iter())
            .map(|(&x, &y)| if x > threshold { y } else { 0. })
            .collect();
        Array::new(data, self.shape.clone())
    }

    pub fn clip(&self, lowerbound: f32, upperbound: f32) -> Array {
        let data = self
            .data
            .iter()
            .map(|&x| x.min(upperbound).max(lowerbound))
            .collect();
        Array::new(data, self.shape.clone())
    }

    pub fn max(&self, axis: usize) -> Array {
        let stride: usize = self.shape[..axis].iter().product();
        let step: usize = if axis >= self.shape.len() {
            1
        } else {
            self.shape[axis + 1..].iter().product()
        };
        let num_summand: usize = self.shape[axis];

        let mut data: Vec<f32> = Vec::new();
        for i in 0..stride {
            for j in 0..step {
                data.push(
                    self.data[step * num_summand * i + j..]
                        .iter()
                        .step_by(step)
                        .take(num_summand)
                        .fold(f32::NEG_INFINITY, |x, y| x.max(*y)),
                )
            }
        }

        let mut shape = self.shape.clone();
        shape[axis] = 1;
        Array::new(data, shape)
    }
}

macro_rules! impl_op {
    ($trait: ident, $fname: ident) => {
        impl $trait for &Array {
            type Output = Array;
            fn $fname(self, rhs: Self) -> Self::Output {
                if self.shape.is_empty() {
                    return self.data[0].$fname(rhs);
                }
                if rhs.shape.is_empty() {
                    return self.$fname(rhs.data[0]);
                }
                if self.shape != rhs.shape {
                    let new_shape =
                        shape_after_broadcast(&self.shape, &rhs.shape).expect(&format!(
                            "Two arrays must have the same shape\nlhs: {:?}\nrhs: {:?}",
                            self, rhs,
                        ));

                    let ldata = self.broadcast_to(&new_shape).data;
                    let rdata = rhs.broadcast_to(&new_shape).data;

                    let data = ldata
                        .iter()
                        .zip(rdata)
                        .map(|(x, y)| f32::$fname(*x, y))
                        .collect();
                    let size = new_shape.iter().product();
                    Array {
                        data,
                        shape: new_shape,
                        size,
                    }
                } else {
                    let data = self
                        .data
                        .iter()
                        .zip(rhs.data.iter())
                        .map(|(x, y)| f32::$fname(*x, y))
                        .collect();
                    Array {
                        data,
                        shape: self.shape.clone(),
                        size: self.size,
                    }
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
                let data = self.data.iter().map(|x| f32::$fname(*x, rhs)).collect();
                Array {
                    data,
                    shape: self.shape.clone(),
                    size: self.size,
                }
            }
        }

        impl $trait<&Array> for f32 {
            type Output = Array;
            fn $fname(self, rhs: &Array) -> Self::Output {
                let data = rhs.data.iter().map(|x| f32::$fname(self, x)).collect();
                Array {
                    data,
                    shape: rhs.shape.clone(),
                    size: rhs.size,
                }
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
        let data = self.data.iter().map(|a| -a).collect();
        Array::new(data, self.shape.clone())
    }
}

impl Neg for &Array {
    type Output = Array;
    fn neg(self) -> Self::Output {
        -self.clone()
    }
}
