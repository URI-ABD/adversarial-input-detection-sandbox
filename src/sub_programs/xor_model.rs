use dfdx::prelude::*;

use crate::model::Model;

const RNG_SEED: u64 = 0;
const EPOCHS: usize = 10000;

const INPUT_COUNT: usize = 2;
const OUTPUT_COUNT: usize = 1;
type NetworkStructure = ((Linear<INPUT_COUNT, 2>, Tanh), Linear<2, OUTPUT_COUNT>);

pub fn xor_model() {
    let mut model: Model<INPUT_COUNT, OUTPUT_COUNT, f64, NetworkStructure> = Model::new(RNG_SEED);

    model.train(
        [
            ([0.0, 0.0], [0.0]),
            ([0.0, 1.0], [1.0]),
            ([1.0, 0.0], [1.0]),
            ([1.0, 1.0], [0.0]),
        ],
        EPOCHS,
        Some(&mut std::io::stdout()),
        None::<std::iter::Empty<_>>, // TODO this is a hacky workaround
    );

    let result = model.predict_batch(&[&[0.0, 0.0], &[0.0, 1.0], &[1.0, 0.0], &[1.0, 1.0]]);

    println!("Result: {result:?}");

    // Interactive prediction
    println!("Interactive prediction!");
    println!(
        "Give two input values between 0.0 and 1.0, and the model's prediction will be shown."
    );
    loop {
        println!(
            "Enter the two input values (space separated), or EXIT to exit the interactive prediction loop:",
        );
        let input = std::io::stdin().lines().next().unwrap().unwrap();

        if input.to_uppercase() == "EXIT" {
            break;
        }

        let nums = input
            .split_whitespace()
            .map(str::parse::<f64>)
            .collect::<Result<Vec<_>, _>>();

        if nums.is_err() || nums.as_ref().unwrap().len() != 2 {
            println!("Invalid input `{input}`");
            continue;
        }

        // Make a prediction with the model
        let prediction = model.predict(&nums.unwrap().try_into().unwrap())[0];

        println!("prediction: {prediction:.4}");
    }
}
