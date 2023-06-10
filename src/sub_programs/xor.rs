use dfdx::{
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
};

const RNG_SEED: u64 = 0;

type Float = f32;
type NetworkStructure = (Linear<2, 2>, Tanh, Linear<2, 1>);
type Model = <NetworkStructure as BuildOnDevice<AutoDevice, Float>>::Built;

fn make_xor_training_data(
    dev: &AutoDevice,
) -> (
    Tensor<(usize, Const<2>), Float, AutoDevice>,
    Tensor<(usize, Const<1>), Float, AutoDevice>,
) {
    (
        dev.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            .realize()
            .unwrap(),
        dev.tensor([[0.0], [1.0], [1.0], [0.0]]).realize().unwrap(),
    )
}

fn train_model(
    model: &mut Model,
    training_data_xs: &Tensor<(usize, Const<2>), Float, AutoDevice>,
    training_data_ys: &Tensor<(usize, Const<1>), Float, AutoDevice>,
    epochs: usize,
) {
    // Allocate gradients for the model (so we don't re-allocate repeatedly)
    let mut gradients = model.alloc_grads();

    // Construct a stochastic gradient descent optimizer for the model
    let mut optimizer: Sgd<Model, Float, AutoDevice> = Sgd::new(
        &model,
        SgdConfig {
            lr: 0.1,                                 // TODO I assume this is short for learning rate?
            momentum: Some(Momentum::Nesterov(0.9)), // TODO I have no idea what this is
            weight_decay: None,                      // TODO what does this mean
        },
    );

    println!(
        "Loss on test data: {:?}",
        evaluate_model(model, &training_data_xs, &training_data_ys),
    );

    // Training loop
    for epoch in 0..epochs {
        // TODO comment this (https://github.com/coreylowman/dfdx/blob/main/examples/05-optim.rs)
        model.zero_grads(&mut gradients);
        let prediction = model.forward_mut(training_data_xs.trace(gradients));
        let loss = mse_loss(prediction, training_data_ys.clone());
        println!("Loss after epoch {epoch}: {:?}", loss.array());
        gradients = loss.backward();
        optimizer.update(model, &gradients).unwrap();
    }
}

fn evaluate_model(
    model: &Model,
    test_data_xs: &Tensor<(usize, Const<2>), Float, AutoDevice>,
    test_data_ys: &Tensor<(usize, Const<1>), Float, AutoDevice>,
) -> Float {
    let logits = model.forward(test_data_xs.clone()); // Why do I need clone here?
    let loss = mse_loss(logits, test_data_ys.clone()); // Why do I need clone here?
    loss.array()
}

pub fn xor() {
    // Create the device (used to allocate and perform computations on tensors)
    let dev: AutoDevice = AutoDevice::seed_from_u64(RNG_SEED);

    // Construct the training data (in this case, just hard-coded XOR data)
    let (xor_data_xs, xor_data_ys): (
        Tensor<(usize, Const<2>), _, _>,
        Tensor<(usize, Const<1>), _, _>,
    ) = make_xor_training_data(&dev);

    // Construct the model (randomly initialized)
    let mut model: Model = dev.build_module::<NetworkStructure, Float>();

    // Train the model
    const EPOCHS: usize = 500;
    train_model(&mut model, &xor_data_xs, &xor_data_ys, EPOCHS);

    // Evaluate the model
    println!(
        "Loss on test data: {:?}",
        evaluate_model(&model, &xor_data_xs, &xor_data_ys),
    );

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
            .map(str::parse::<f32>)
            .collect::<Result<Vec<_>, _>>();

        if nums.is_err() || nums.as_ref().unwrap().len() != 2 {
            println!("Invalid input `{input}`");
            continue;
        }

        let input: Tensor<(Const<1>, Const<2>), Float, AutoDevice> =
            dev.tensor([TryInto::<[Float; 2]>::try_into(nums.unwrap()).unwrap()]);

        // Make a prediction with the model
        let prediction = model.forward(input).array()[0][0];

        println!("prediction: {prediction:.4}");
    }
}
