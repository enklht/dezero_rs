use super::Array;

impl Array {
    pub fn all_close(&self, rhs: &Array, tol: f32) -> bool {
        (self - rhs)
            .data
            .iter()
            .try_for_each(|x| if x < &tol { Some(()) } else { None })
            .is_some()
    }

    pub fn size(&self) -> usize {
        self.size
    }
}

macro_rules! inner {
    ($x: expr, $y: expr) => {
        $x.zip($y).fold(0., |acc, (x, y)| acc + x * y)
    };
}

pub(super) fn shape_after_broadcast(shape0: &[usize], shape1: &[usize]) -> Option<Vec<usize>> {
    let mut res = Vec::new();
    if shape0.len() <= shape1.len() {
        for (&n, &m) in std::iter::repeat_n(&1, shape1.len() - shape0.len())
            .chain(shape0.iter())
            .zip(shape1.iter())
        {
            match (n, m) {
                (1, m) => res.push(m),
                (n, 1) => res.push(n),
                (n, m) if n == m => res.push(m),
                _ => return None,
            }
        }
    } else {
        for (&n, &m) in std::iter::repeat_n(&1, shape0.len() - shape1.len())
            .chain(shape1.iter())
            .zip(shape0.iter())
        {
            match (n, m) {
                (1, m) => res.push(m),
                (n, 1) => res.push(n),
                (n, m) if n == m => res.push(m),
                _ => return None,
            }
        }
    }
    Some(res)
}

pub(super) fn broadcast_to(data: &[f32], old_shape: &[usize], new_shape: &[usize]) -> Vec<f32> {
    let mut axes = Vec::new();
    let mut dups = Vec::new();
    for (axis, (i, j)) in old_shape.iter().zip(new_shape.iter()).enumerate() {
        match (i, j) {
            (i, j) if i == j => {}
            (1, &dup) => {
                axes.push(axis);
                dups.push(dup);
            }
            _ => panic!("failed to broadcast {:?} to {:?}", data, new_shape),
        }
    }

    let mut data = data.to_vec();
    let dim = new_shape.len();

    let mut chunk_sizes = vec![1];
    for x in new_shape.iter().rev() {
        chunk_sizes.push(chunk_sizes.last().unwrap() * x);
    }

    for (&axis, &dup) in axes.iter().zip(dups.iter()) {
        data = data
            .chunks(chunk_sizes[dim - axis - 1])
            .flat_map(|c| c.repeat(dup))
            .collect();
    }
    data
}

pub(super) fn matmul_2d(lhs: &[f32], rhs: &[f32], (l, m, n): (usize, usize, usize)) -> Vec<f32> {
    let mut data = Vec::with_capacity(l * n);
    for i in 0..l {
        for j in 0..n {
            data.push(inner!(
                lhs[m * i..].iter().take(m),
                rhs[j..].iter().step_by(n)
            ));
        }
    }
    data
}
