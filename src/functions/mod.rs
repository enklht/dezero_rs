pub mod aggregate;
pub mod binary;
pub mod call;
pub mod call_impl;
pub mod funcbox;
pub mod macros;
pub mod neural;
pub mod trait_;
pub mod unary;

pub use self::aggregate::*;
pub use self::binary::*;
pub use self::call::{cross_entropy_loss, linear, mean_squared_error, relu, sigmoid, softmax};
pub use self::call_impl::call;
pub use self::funcbox::FuncBox;
pub use self::neural::{CrossEntropy, Linear, MeanSquaredError, ReLU, Sigmoid, Softmax};
pub use self::trait_::Function;
pub use self::unary::*;
