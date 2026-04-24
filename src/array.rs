mod macros;
mod ops;
mod utils;

use ndarray::ArrayD;
use rand::{distributions::Standard, Rng};
use rand_distr::{Distribution, Normal};
use std::fmt::Display;

#[derive(Debug, Clone, PartialEq)]
pub struct Array {
    data: ArrayD<f32>,
}

impl Array {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Array {
        let size: usize = shape.iter().product();
        if data.len() != size {
            panic!("The data and the shape are inconsistent")
        }
        let data = ArrayD::from_shape_vec(shape, data).expect("Invalid shape");
        Array { data }
    }

    pub fn read_csv(path: &str) -> Array {
        let f = std::fs::read_to_string(path).expect("File not found");
        let lines = f.lines().collect::<Vec<_>>();
        let num_rows = lines.len();
        let num_cols = lines[0].split(',').count();
        let data = lines
            .iter()
            .flat_map(|s| s.split(',').map(|d| d.parse::<f32>().unwrap()))
            .collect::<Vec<f32>>();

        Array::new(data, vec![num_rows, num_cols])
    }

    pub fn write_csv(&self, path: &str) {
        if self.data.ndim() != 2 {
            panic!("This implementation is temporary and can only handle 2dim arrays.")
        }
        let shape = self.data.shape();
        let cols = shape[1];
        let data = self.data.as_slice().expect("Array not contiguous");
        let string = data
            .chunks(cols)
            .map(|row| {
                row.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<String>>()
                    .join(",")
            })
            .collect::<Vec<_>>()
            .join("\n");

        std::fs::write(path, string).unwrap();
    }

    pub fn zeros(shape: &[usize]) -> Array {
        let size: usize = shape.iter().product();
        let data = vec![0.; size];
        Array::new(data, shape.to_vec())
    }

    pub fn ones(shape: &[usize]) -> Array {
        let size: usize = shape.iter().product();
        let data = vec![1.; size];
        Array::new(data, shape.to_vec())
    }

    pub fn rand(shape: &[usize]) -> Array {
        let size: usize = shape.iter().product();
        let rng = rand::thread_rng();
        let data = rng.sample_iter(Standard).take(size).collect();
        Array::new(data, shape.to_vec())
    }

    pub fn randn(shape: &[usize], mean: f32, std_dev: f32) -> Array {
        let size: usize = shape.iter().product();
        let rng = rand::thread_rng();
        let normal = Normal::new(mean, std_dev).unwrap();
        let data = normal.sample_iter(rng).take(size).collect();
        Array::new(data, shape.to_vec())
    }

    pub fn get_data(&self) -> &[f32] {
        self.data.as_slice().expect("Array not contiguous")
    }

    pub fn get_shape(&self) -> &[usize] {
        self.data.shape()
    }

    pub fn set_data(&mut self, new: Array) {
        let shape = self.data.shape().to_vec();
        if new.data.shape() != shape.as_slice() {
            panic!("The data and the shape are inconsistent")
        }
        let data = self.data.as_slice_mut().expect("Array not contiguous");
        let new_data = new.data.as_slice().expect("Array not contiguous");
        for (old, new) in data.iter_mut().zip(new_data.iter()) {
            *old = *new
        }
    }

    pub fn to_string(&self, depth: usize) -> String {
        let data = self.data.as_slice().expect("Array not contiguous");
        array_to_string(data, self.data.shape(), depth)
    }
}

impl Display for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let data = self.data.as_slice().expect("Array not contiguous");
        let string = array_to_string(data, self.data.shape(), 0);
        write!(f, "{}", string)
    }
}

fn array_to_string(data: &[f32], shape: &[usize], depth: usize) -> String {
    match shape.len() {
        0 => data[0].to_string(),
        1 => format!("{:?}", data),
        _ => {
            let mut acc = "[".to_string();
            acc += &data
                .chunks(data.len() / shape[0])
                .map(|row| array_to_string(row, &shape[1..], depth + 1))
                .collect::<Vec<_>>()
                .join(&format!("\n{}", " ".repeat(depth + 1)));
            acc + "]"
        }
    }
}
