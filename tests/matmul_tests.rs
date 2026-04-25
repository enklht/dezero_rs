extern crate dezero;

use dezero::{array::Array, array0, array1, array_with_shape};

#[test]
fn matmul_2d_matrix_multiply() {
    let x = array1!(0..16).reshape(&[8, 2]);
    let y = array1!(0..4).reshape(&[2, 2]);

    assert_eq!(
        x.matmul(&y),
        array1!([2, 3, 6, 11, 10, 19, 14, 27, 18, 35, 22, 43, 26, 51, 30, 59]).reshape(&[8, 2])
    );
}

#[test]
fn matmul_batched_matrix_multiply() {
    let x = array1!(0..16).reshape(&[2, 4, 2]);
    let y = array1!(0..4).reshape(&[2, 2]);

    assert_eq!(
        x.matmul(&y),
        array1!([2, 3, 6, 11, 10, 19, 14, 27, 18, 35, 22, 43, 26, 51, 30, 59]).reshape(&[2, 4, 2])
    );
}

#[test]
fn matmul_vector_dot_product() {
    let x = array1!([1, 2, 3, 4, 5]);
    assert_eq!(x.matmul(&x), array0!(55));
}

#[test]
fn matmul_broadcasted_shape() {
    let a = Array::ones(&[9, 5, 7, 4]);
    let c = Array::ones(&[9, 5, 4, 3]);

    assert_eq!(a.matmul(&c).shape(), &[9, 5, 7, 3])
}

#[test]
fn matmul5_broadcast_stack_order() {
    let lhs = array1!(0..8).reshape(&[2, 1, 2, 2]);
    let rhs = array1!(0..12).reshape(&[1, 3, 2, 2]);

    assert_eq!(lhs.matmul(&rhs).shape(), &[2, 3, 2, 2]);
    assert_eq!(
        lhs.matmul(&rhs),
        array_with_shape!(
            [
                2., 3., 6., 11., 6., 7., 26., 31., 10., 11., 46., 51., 10., 19., 14., 27., 46.,
                55., 66., 79., 82., 91., 118., 131.
            ],
            [2, 3, 2, 2]
        )
    );
}

#[test]
fn matmul6_vector_matrix_shape() {
    let v = array1!([1., 2., 3.]);
    let m = array_with_shape!([1., 2., 3., 4., 5., 6.], [3, 2]);

    assert_eq!(v.matmul(&m).shape(), &[2]);
    assert_eq!(v.matmul(&m), array1!([22., 28.]));
}

#[test]
fn matmul7_matrix_vector_shape() {
    let m = array_with_shape!([1., 2., 3., 4., 5., 6.], [2, 3]);
    let v = array1!([1., 2., 3.]);

    assert_eq!(m.matmul(&v).shape(), &[2]);
    assert_eq!(m.matmul(&v), array1!([14., 32.]));
}
