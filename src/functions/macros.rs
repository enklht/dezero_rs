macro_rules! impl_getters_setters {
    () => {
        fn get_generation(&self) -> u32 {
            self.generation
        }

        fn get_inputs(&self) -> Vec<VBox> {
            self.inputs.clone().unwrap()
        }

        fn get_output(&self) -> WeakVBox {
            self.output.clone().unwrap()
        }

        fn set_generation(&mut self, gen: u32) {
            self.generation = gen
        }

        fn set_inputs(&mut self, inputs: Vec<VBox>) {
            self.inputs = Some(inputs)
        }

        fn set_output(&mut self, output: WeakVBox) {
            self.output = Some(output)
        }
    };
}

macro_rules! define {
    ($name: ident) => {
        pub struct $name {
            inputs: Option<Vec<VBox>>,
            output: Option<WeakVBox>,
            generation: u32,
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $name {
            pub fn new() -> Self {
                Self {
                    inputs: None,
                    output: None,
                    generation: 0,
                }
            }
        }
    };
    ($name: ident, $($key: ident: $type: ty),*) => {
        pub struct $name {
            inputs: Option<Vec<VBox>>,
            output: Option<WeakVBox>,
            generation: u32,
            $(
                $key: $type
            ),*
        }

        impl $name {
            pub fn new($($key: $type),*) -> Self {
                Self {
                    inputs: None,
                    output: None,
                    generation: 0,
                    $(
                        $key
                    ),*
                }
            }
        }
    };
}

pub(super) use define;
pub(super) use impl_getters_setters;
