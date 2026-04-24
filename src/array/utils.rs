use super::Array;

impl Array {
    pub fn all_close(&self, rhs: &Array, tol: f32) -> bool {
        (self - rhs)
            .data
            .iter()
            .try_for_each(|x| if x.abs() < tol { Some(()) } else { None })
            .is_some()
    }

    pub fn size(&self) -> usize {
        self.data.len()
    }
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
