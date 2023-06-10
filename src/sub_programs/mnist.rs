use dfdx::prelude::*;
use mnist::{MnistBuilder, NormalizedMnist};

use crate::model::Model;

const RNG_SEED: u64 = 0;
const EPOCHS: usize = 1500;

const INPUT_COUNT: usize = 28 * 28;
const OUTPUT_COUNT: usize = 10;
type NetworkStructure = (
    (Linear<INPUT_COUNT, 512>, ReLU),
    (Linear<512, 128>, ReLU),
    (Linear<128, 32>, ReLU),
    Linear<32, OUTPUT_COUNT>,
);

#[derive(Debug)]
struct MnistData {
    training_images: Vec<[f32; INPUT_COUNT]>,
    training_labels: Vec<[f32; OUTPUT_COUNT]>,
    validation_images: Vec<[f32; INPUT_COUNT]>,
    validation_labels: Vec<[f32; OUTPUT_COUNT]>,
    testing_images: Vec<[f32; INPUT_COUNT]>,
    testing_labels: Vec<[f32; OUTPUT_COUNT]>,
}

fn load_mnist_data() -> MnistData {
    // TODO do something with the test dataset

    // TODO get this from the user
    const MNIST_PATH: &'static str = "data/datasets/MNIST_digits";

    let NormalizedMnist {
        trn_img,
        trn_lbl,
        val_img,
        val_lbl,
        tst_img,
        tst_lbl,
    } = MnistBuilder::new()
        .base_path(MNIST_PATH)
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .finalize()
        .normalize();

    assert_eq!(trn_img.len(), 50_000 * INPUT_COUNT);
    assert_eq!(trn_lbl.len(), 50_000);
    assert_eq!(val_img.len(), 10_000 * INPUT_COUNT);
    assert_eq!(val_lbl.len(), 10_000);
    assert_eq!(tst_img.len(), 10_000 * INPUT_COUNT);
    assert_eq!(tst_lbl.len(), 10_000);

    let (training_images, training_labels) = (0..50_000)
        .map(|i| {
            let input: [f32; INPUT_COUNT] = trn_img[i * INPUT_COUNT..(i + 1) * INPUT_COUNT]
                .try_into()
                .unwrap();
            let mut output: [f32; OUTPUT_COUNT] = [0.0; OUTPUT_COUNT];
            output[trn_lbl[i] as usize] = 1.0;

            (input, output)
        })
        .unzip::<[f32; INPUT_COUNT], [f32; OUTPUT_COUNT], Vec<_>, Vec<_>>();
    let (training_images, training_labels) = (
        training_images.try_into().unwrap(),
        training_labels.try_into().unwrap(),
    );

    let (validation_images, validation_labels) = (0..10_000)
        .map(|i| {
            let input: [f32; INPUT_COUNT] = val_img[i * INPUT_COUNT..(i + 1) * INPUT_COUNT]
                .try_into()
                .unwrap();
            let mut output: [f32; OUTPUT_COUNT] = [0.0; OUTPUT_COUNT];
            output[val_lbl[i] as usize] = 1.0;

            (input, output)
        })
        .unzip::<[f32; INPUT_COUNT], [f32; OUTPUT_COUNT], Vec<_>, Vec<_>>();
    let (validation_images, validation_labels) = (
        validation_images.try_into().unwrap(),
        validation_labels.try_into().unwrap(),
    );

    let (testing_images, testing_labels) = (0..10_000)
        .map(|i| {
            let input: [f32; INPUT_COUNT] = tst_img[i * INPUT_COUNT..(i + 1) * INPUT_COUNT]
                .try_into()
                .unwrap();
            let mut output: [f32; OUTPUT_COUNT] = [0.0; OUTPUT_COUNT];
            output[tst_lbl[i] as usize] = 1.0;

            (input, output)
        })
        .unzip::<[f32; INPUT_COUNT], [f32; OUTPUT_COUNT], Vec<_>, Vec<_>>();
    let (testing_images, testing_labels) = (
        testing_images.try_into().unwrap(),
        testing_labels.try_into().unwrap(),
    );

    MnistData {
        training_images,
        training_labels,
        validation_images,
        validation_labels,
        testing_images,
        testing_labels,
    }
}

pub fn mnist() {
    let mut model: Model<INPUT_COUNT, OUTPUT_COUNT, f32, NetworkStructure> = Model::new(RNG_SEED);

    println!("Loading MNIST dataset...");
    let data = load_mnist_data();
    println!("Dataset loaded.");

    println!("Training model...");
    model.train(
        data.training_images
            .into_iter()
            .zip(data.training_labels.into_iter()),
        EPOCHS,
        Some(&mut std::io::stdout()),
        Some(
            data.validation_images
                .into_iter()
                .zip(data.validation_labels.into_iter()),
        ),
    );
    println!("Model trained.");

    // Evaluate model accuracy
    let testing_loss = model.evaluate_loss(
        data.testing_images
            .into_iter()
            .zip(data.testing_labels.into_iter()),
    );
    println!("Model testing loss: {testing_loss}");

    // // TODO temp
    // model
    //     .model
    //     .save("data/trained_models/trained_mnist_model_50000_epochs.npz")
    //     .unwrap();

    // Interactive prediction
    // TODO
    println!("Interactive prediction not yet implemented");
}
