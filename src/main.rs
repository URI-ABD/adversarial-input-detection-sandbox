#![warn(missing_debug_implementations, unsafe_code)]
#![deny(unsafe_op_in_unsafe_fn)]

use std::fmt::Display;

mod cli;
mod model;
mod sub_programs;

choices_enum! {
    enum MainActions {
        Exit as "Exit",
        TrainAndEvalXor as "Train and evaluate a model on XOR",
        XorModel as "Train and evaluate a model on XOR (using the fancy new Model abstraction)",
        TrainAndEvalMnist as "Train and evaluate a model on MNIST handwritten digit classification",
    }
}

fn main() {
    loop {
        use MainActions::*;
        match cli::prompt_choice("What would you like to do?", MainActions::all()) {
            None | Some(Exit) => break,
            Some(TrainAndEvalXor) => sub_programs::xor(),
            Some(XorModel) => sub_programs::xor_model(),
            Some(TrainAndEvalMnist) => sub_programs::mnist(),
        }
    }
}
