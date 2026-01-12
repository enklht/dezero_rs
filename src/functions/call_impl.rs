use crate::{functions::funcbox::FuncBox, functions::trait_::Function, variable::VBox};
use std::rc::Rc;

pub fn call(mut f: impl Function + 'static, input: &[VBox]) -> VBox {
    let x = input.iter().map(|i| i.get_array()).collect();
    let y = f.forward(x);
    let output = VBox::new(y);
    f.set_inputs(input.into());
    f.set_output(output.clone().downgrade());

    if *crate::ENABLE_BACKPROP.lock().unwrap() {
        f.set_generation(input.iter().map(|x| x.get_gen()).max().unwrap());
        let to_f = Rc::new(f);
        output.set_creator(FuncBox(to_f.clone()));
    }
    output
}
