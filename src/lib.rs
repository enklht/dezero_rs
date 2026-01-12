pub mod array;
pub mod functions;
pub mod layers;
pub mod macros;
pub mod optimizers;
pub mod variable;

use std::sync::Mutex;

pub static ENABLE_BACKPROP: Mutex<bool> = Mutex::new(true);
