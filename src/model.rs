use dfdx::{optim::*, prelude::*};

/// A machine learning model, capable of being trained and making predictions
///
/// TODO make this generic over device (not generic enough already, lol)
pub struct Model<
    const INPUT_COUNT: usize,
    const OUTPUT_COUNT: usize,
    DataType: Dtype,
    NetworkStructure,
> where
    AutoDevice: Device<DataType>,
    NetworkStructure: BuildOnDevice<AutoDevice, DataType>,
{
    model: <NetworkStructure as BuildOnDevice<AutoDevice, DataType>>::Built,

    // TODO can we/do we want to only store a reference to this? Or not store it
    // at all?
    device: AutoDevice,
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize, DataType: Dtype, NetworkStructure>
    Model<INPUT_COUNT, OUTPUT_COUNT, DataType, NetworkStructure>
where
    AutoDevice: Device<DataType>,
    NetworkStructure: BuildOnDevice<AutoDevice, DataType>,
{
    /// Constructs a new `Model` with the given RNG seed
    pub fn new(seed: u64) -> Self {
        let device = AutoDevice::seed_from_u64(seed);
        let model = device.build_module::<NetworkStructure, DataType>();

        Self { device, model }
    }
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize, DataType: Dtype, NetworkStructure>
    Model<INPUT_COUNT, OUTPUT_COUNT, DataType, NetworkStructure>
where
    AutoDevice: Device<DataType>,
    NetworkStructure: BuildOnDevice<AutoDevice, DataType>,
    <NetworkStructure as BuildOnDevice<AutoDevice, DataType>>::Built: Module<
            Tensor<Rank1<INPUT_COUNT>, DataType, AutoDevice>,
            Output = Tensor<Rank1<OUTPUT_COUNT>, DataType, AutoDevice>,
        > + Module<
            Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice>,
            Output = Tensor<(usize, Const<OUTPUT_COUNT>), DataType, AutoDevice>,
        >,
{
    /// Makes a prediction for the given input data
    pub fn predict(&self, input: &[DataType; INPUT_COUNT]) -> [DataType; OUTPUT_COUNT] {
        let input_tensor: Tensor<Rank1<INPUT_COUNT>, DataType, AutoDevice> =
            self.device.tensor(input);

        let output_tensor: Tensor<Rank1<OUTPUT_COUNT>, DataType, AutoDevice> =
            self.model.forward(input_tensor);

        let output: [DataType; OUTPUT_COUNT] = output_tensor.array();

        output
    }

    /// Makes predictions for the given input data
    pub fn predict_batch(
        &self,
        input: &[&[DataType; INPUT_COUNT]],
    ) -> Vec<[DataType; OUTPUT_COUNT]> {
        let input_tensor: Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice> =
            self.device.tensor((
                input // TODO this is gross, gotta be a cleaner way
                    .iter()
                    .flat_map(|slice| slice.iter())
                    .copied()
                    .collect::<Vec<DataType>>(),
                (input.len(), Const::<INPUT_COUNT>),
            ));

        let output_tensor: Tensor<(usize, Const<OUTPUT_COUNT>), DataType, AutoDevice> =
            self.model.forward(input_tensor);

        let output: Vec<[DataType; OUTPUT_COUNT]> = (0..input.len())
            .map(|index| {
                output_tensor
                    .clone() // TODO not clone here
                    .select(self.device.tensor(index))
                    .array()
            })
            .collect();

        output
    }

    /// Evaluates the model's loss on a given test dataset
    pub fn evaluate_loss<
        TestingDataIterator: IntoIterator<Item = ([DataType; INPUT_COUNT], [DataType; OUTPUT_COUNT])>,
    >(
        &self,
        testing_data: TestingDataIterator,
    ) -> DataType {
        let (testing_data_inputs_vec, testing_data_outputs_vec): (
            Vec<[DataType; INPUT_COUNT]>,
            Vec<[DataType; OUTPUT_COUNT]>,
        ) = testing_data.into_iter().unzip();
        let testing_data_len = testing_data_inputs_vec.len();
        let testing_data_inputs: Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice> =
            self.device.tensor((
                testing_data_inputs_vec
                    .into_iter()
                    .flatten()
                    .collect::<Vec<DataType>>(),
                (testing_data_len, Const::<INPUT_COUNT>),
            ));
        let testing_data_outputs: Tensor<(usize, Const<OUTPUT_COUNT>), DataType, AutoDevice> =
            self.device.tensor((
                testing_data_outputs_vec
                    .into_iter()
                    .flatten()
                    .collect::<Vec<DataType>>(),
                (testing_data_len, Const::<OUTPUT_COUNT>),
            ));

        let test_prediction = self.model.forward(testing_data_inputs);
        // TODO Why mse as opposed to some other loss function?
        let test_loss = mse_loss(test_prediction, testing_data_outputs).array();

        test_loss
    }
}

impl<const INPUT_COUNT: usize, const OUTPUT_COUNT: usize, DataType: Dtype, NetworkStructure>
    Model<INPUT_COUNT, OUTPUT_COUNT, DataType, NetworkStructure>
where
    AutoDevice: Device<DataType>,
    NetworkStructure: BuildOnDevice<AutoDevice, DataType>,
    <NetworkStructure as BuildOnDevice<AutoDevice, DataType>>::Built: Module<
        Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice>,
        Output = Tensor<(usize, Const<OUTPUT_COUNT>), DataType, AutoDevice>,
    >,
    <NetworkStructure as BuildOnDevice<AutoDevice, DataType>>::Built: ModuleMut<
        Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice, OwnedTape<DataType, AutoDevice>>,
        Output = Tensor<
            (usize, Const<OUTPUT_COUNT>),
            DataType,
            AutoDevice,
            OwnedTape<DataType, AutoDevice>,
        >,
    >,
{
    /// Trains the model
    ///
    /// # Panics
    /// - If writing to the given `output` fails
    /// - If something goes wrong with the optimizer or device
    ///
    /// # Arguments
    /// - `training_data` - an iterator of training data
    /// - `epochs` - the number of learning iterations
    /// - `output` - an optional writer for progress updates
    pub fn train<
        TrainingDataIterator: IntoIterator<Item = ([DataType; INPUT_COUNT], [DataType; OUTPUT_COUNT])>,
        Writer: std::io::Write,
        TestingDataIterator: IntoIterator<Item = ([DataType; INPUT_COUNT], [DataType; OUTPUT_COUNT])>,
    >(
        &mut self,
        training_data: TrainingDataIterator,
        epochs: usize,
        mut output: Option<&mut Writer>,
        testing_data: Option<TestingDataIterator>,
    ) {
        // TODO "batch"-ize this

        // Convert the training and testing data to tensors
        // TODO look into a "proper" way to do this
        let (training_data_inputs_vec, training_data_outputs_vec): (
            Vec<[DataType; INPUT_COUNT]>,
            Vec<[DataType; OUTPUT_COUNT]>,
        ) = training_data.into_iter().unzip();
        let training_data_len = training_data_inputs_vec.len();
        let training_data_inputs: Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice> =
            self.device.tensor((
                training_data_inputs_vec
                    .into_iter()
                    .flatten()
                    .collect::<Vec<DataType>>(),
                (training_data_len, Const::<INPUT_COUNT>),
            ));
        let training_data_outputs: Tensor<(usize, Const<OUTPUT_COUNT>), DataType, AutoDevice> =
            self.device.tensor((
                training_data_outputs_vec
                    .into_iter()
                    .flatten()
                    .collect::<Vec<DataType>>(),
                (training_data_len, Const::<OUTPUT_COUNT>),
            ));
        let (testing_data_inputs_vec, testing_data_outputs_vec): (
            Option<Vec<[DataType; INPUT_COUNT]>>,
            Option<Vec<[DataType; OUTPUT_COUNT]>>,
        ) = match testing_data {
            Some(testing_data) => {
                let (inputs, outputs) = testing_data.into_iter().unzip();
                (Some(inputs), Some(outputs))
            }
            None => (None, None),
        };
        let testing_data_len = testing_data_inputs_vec.as_ref().map(|inputs| inputs.len());
        let testing_data_inputs: Option<Tensor<(usize, Const<INPUT_COUNT>), DataType, AutoDevice>> =
            match testing_data_inputs_vec {
                Some(testing_data_inputs_vec) => Some(
                    self.device.tensor((
                        testing_data_inputs_vec
                            .into_iter()
                            .flatten()
                            .collect::<Vec<DataType>>(),
                        (testing_data_len.unwrap(), Const::<INPUT_COUNT>),
                    )),
                ),
                None => None,
            };
        let testing_data_outputs: Option<
            Tensor<(usize, Const<OUTPUT_COUNT>), DataType, AutoDevice>,
        > = match testing_data_outputs_vec {
            Some(testing_data_outputs_vec) => Some(
                self.device.tensor((
                    testing_data_outputs_vec
                        .into_iter()
                        .flatten()
                        .collect::<Vec<DataType>>(),
                    (testing_data_len.unwrap(), Const::<OUTPUT_COUNT>),
                )),
            ),
            None => None,
        };

        // Allocate gradients for the model
        let mut gradients =
            <<NetworkStructure as BuildOnDevice<AutoDevice, DataType>>::Built as ZeroGrads<
                DataType,
                AutoDevice,
            >>::alloc_grads(&self.model);

        // Construct a stochastic gradient descent optimizer for the model
        // TODO allow custom optimizers, or at LEAST custom hyperparameters
        // let mut optimizer: Sgd<_, DataType, AutoDevice> = Sgd::new(
        //     &self.model,
        //     SgdConfig {
        //         lr: DataType::from_f64(0.1).unwrap(),
        //         momentum: Some(Momentum::Nesterov(DataType::from_f64(0.9).unwrap())), // TODO I have no idea what this is
        //         ..Default::default()
        //     }, // TODO What hyperparameters do we want?
        // );
        let mut optimizer = Adam::new(&self.model, AdamConfig::default());

        // Training loop
        for epoch in 1..=epochs {
            // Zero out the gradients
            self.model.zero_grads(&mut gradients);

            // Make a prediction (uses forward_mut since this is for training)
            let prediction = self
                .model
                .forward_mut(training_data_inputs.trace(gradients));

            // Compute the mean squared error loss of the prediction
            // TODO Why mse as opposed to some other loss function?
            // TODO Can we avoid the clone here? It has it in the example too... is it cheap?
            let loss = mse_loss(prediction, training_data_outputs.clone());

            // Epoch progress update
            if let Some(output) = output.as_mut() {
                writeln!(
                    output,
                    "Training loss after epoch {epoch}: {:?}",
                    loss.array()
                )
                .unwrap();

                // Evaluate the model on the test data
                if testing_data_inputs.is_some() {
                    let test_prediction = self.model.forward(testing_data_inputs.clone().unwrap());
                    // TODO Why mse as opposed to some other loss function?
                    // TODO Can we avoid the clone here? It has it in the example too... is it cheap?
                    let test_loss =
                        mse_loss(test_prediction, testing_data_outputs.clone().unwrap());
                    writeln!(
                        output,
                        "Testing loss after epoch {epoch}:  {:?}",
                        test_loss.array()
                    )
                    .unwrap();
                    writeln!(output).unwrap();
                }
            }

            // Compute the optimal gradients with back-propogation and update
            // the model with our optimizer
            gradients = loss.backward();
            optimizer.update(&mut self.model, &gradients).unwrap();
        }
    }
}
