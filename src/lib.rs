#[no_mangle]
fn matadd(
    matrix_a: &Vec<Vec<f64>>,
    matrix_b: &Vec<Vec<f64>>,
    _rows: usize,
    _cols: usize,
) -> Vec<Vec<f64>> {
    matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b.iter())
                .map(|(&a, &b)| a + b)
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[no_mangle]
fn matsub(
    matrix_a: &Vec<Vec<f64>>,
    matrix_b: &Vec<Vec<f64>>,
    _rows: usize,
    _cols: usize,
) -> Vec<Vec<f64>> {
    matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b.iter())
                .map(|(&a, &b)| a - b)
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[no_mangle]
fn matsub_lr(
    matrix_a: &Vec<Vec<f64>>,
    matrix_b: &Vec<Vec<f64>>,
    _rows: usize,
    _cols: usize,
    lr: f64,
) -> Vec<Vec<f64>> {
    matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b.iter())
                .map(|(&a, &b)| a - lr * b)
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[no_mangle]
fn pairwisemul(
    matrix_a: &Vec<Vec<f64>>,
    matrix_b: &Vec<Vec<f64>>,
    _rows: usize,
    _cols: usize,
) -> Vec<Vec<f64>> {
    matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b.iter())
                .map(|(&a, &b)| a * b)
                .collect::<Vec<f64>>()
        })
        .collect()
}
#[no_mangle]
fn matmul(
    matrix_a: &Vec<Vec<f64>>,
    rows_a: usize,
    cols_a: usize,
    matrix_b: &Vec<Vec<f64>>,
    rows_b: usize,
    cols_b: usize,
) -> Option<Vec<Vec<f64>>> {
    if cols_a != rows_b {
        return None; // Matrix dimensions are incompatible for multiplication
    }

    let mut result = vec![vec![0.0; cols_b]; rows_a];

    for i in 0..rows_a {
        for j in 0..cols_b {
            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += matrix_a[i][k] * matrix_b[k][j];
            }
            result[i][j] = sum;
        }
    }

    Some(result)
}

pub trait Layer {
    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn backward(&mut self, output_errors: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
}

#[derive(Clone, Copy, Debug)]
enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
    Softmax = 3,
    BinaryActivation = 4,
}

impl Activation {
    fn from_int(value: usize) -> Option<Activation> {
        match value {
            0 => Some(Activation::Tanh),
            1 => Some(Activation::Sigmoid),
            2 => Some(Activation::Relu),
            3 => Some(Activation::Softmax),
            4 => Some(Activation::BinaryActivation),
            _ => None,
        }
    }
}
#[no_mangle]
fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let rows = matrix.len();
    let cols = matrix[0].len();

    let mut result: Vec<Vec<f64>> = vec![vec![0.0; rows]; cols];

    for (i, row) in matrix.into_iter().enumerate() {
        for (j, value) in row.into_iter().enumerate() {
            result[j][i] = *value;
        }
    }

    result
}
#[no_mangle]
fn apply_activation(
    vector: &Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
    activation: Activation,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            match activation {
                Activation::Tanh => {
                    result[i][j] = vector[i][j].tanh();
                }
                Activation::Sigmoid => {
                    result[i][j] = 1.0 / (1.0 + (-vector[i][j]).exp());
                }
                Activation::Relu => {
                    result[i][j] = vector[i][j].max(0.0);
                }
                Activation::Softmax => {
                    let max_value = vector[i].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let exp_sum: f64 = vector[i].iter().map(|&x| (x - max_value).exp()).sum();
                    result[i][j] = (vector[i][j] - max_value).exp() / exp_sum;
                }
                Activation::BinaryActivation => {
                    result[i][j] = if vector[i][j] > 0.5 { 1.0 } else { 0.0 };
                }
            }
        }
    }

    result
}

#[no_mangle]
fn activation_derivative(
    vector: &Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
    activation: Activation,
) -> Vec<Vec<f64>> {
    let mut result = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            match activation {
                Activation::Tanh => {
                    result[i][j] = 1.0 - vector[i][j].tanh().powi(2);
                }
                Activation::Sigmoid => {
                    let sigmoid = 1.0 / (1.0 + (-vector[i][j]).exp());
                    result[i][j] = sigmoid * (1.0 - sigmoid);
                }
                Activation::Relu => {
                    result[i][j] = if vector[i][j] > 0.0 { 1.0 } else { 0.0 };
                }
                Activation::Softmax | _=> {
                    panic!("Derivative of softmax function is not defined.");
                }
            }
        }
    }

    result
}

pub struct DenseLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<Vec<f64>>,
    inputs: Vec<Vec<f64>>,
    outputs: Vec<Vec<f64>>,
    input_errors: Vec<Vec<f64>>,
    output_errors: Vec<Vec<f64>>,
    activations: Vec<Vec<f64>>,
    input_size: usize,
    output_size: usize,
    lr: f64,
    activation: Activation,
}

impl DenseLayer {
    // weights is an array of input_size rows and output_size columns
    // i-th row j-th column is the weight connecting i-th output neuron to j-th input neuron
    pub fn new(input_size: usize, output_size: usize, lr: f64, activation: usize) -> DenseLayer {
        DenseLayer {
            input_size,
            output_size,
            lr,
            input_errors: vec![vec![0.0; input_size]],
            output_errors: vec![vec![0.0; output_size]],
            inputs: vec![vec![0.0; input_size]],
            outputs: vec![vec![0.0; output_size]],
            activations: vec![vec![0.0; output_size]],
            activation: Activation::from_int(activation).unwrap(),
            weights: vec![vec![0.0; output_size]; input_size],
            biases: vec![vec![0.0; output_size]],
        }
    }
}
// input: 1x10
// output: 1x2
// weights: 10x2

impl Layer for DenseLayer {
    fn forward(&mut self, _input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.activations = matadd(
            &matmul(
                &self.inputs,
                1,
                self.input_size,
                &self.weights,
                self.input_size,
                self.output_size,
            )
            .unwrap(),
            &self.biases,
            1,
            self.input_size,
        );
        self.outputs = apply_activation(&self.activations, 1, self.output_size, self.activation);
        self.outputs.clone()
    }
    // self.inputs: 1x256
    // self.weights:
    fn backward(&mut self, output_errors: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        self.output_errors = output_errors.clone();
        // [IS,OS]x[OS,1] = [IS,1]
        let intermediate = transpose(
            &matmul(
                &self.weights,
                self.input_size,
                self.output_size,
                &transpose(&output_errors),
                self.output_size,
                1,
            )
            .unwrap(),
        );

        self.input_errors = pairwisemul(
            &activation_derivative(&self.inputs, 1, self.output_size, self.activation),
            &intermediate,
            1,
            self.input_size,
        );

        let weight_errors = matmul(
            &transpose(&self.inputs),
            1,
            self.input_size,
            &output_errors,
            1,
            self.output_size,
        )
        .unwrap();
        self.weights = matsub_lr(
            &self.weights,
            &weight_errors,
            self.input_size,
            self.output_size,
            self.lr,
        );
        self.biases = matsub_lr(&self.biases, &output_errors, 1, self.output_size, self.lr);
        self.input_errors.clone()
    }
}

pub struct NeuralNetwork {
    layers: Vec<Box<dyn Layer>>,
    loss: Vec<f64>,
}
#[no_mangle]
pub fn mean_square_error(expected: &Vec<Vec<f64>>, predicted: &Vec<Vec<f64>>) -> f64 {
    let mut mse = 0.0;

    for (exp_row, pred_row) in expected.iter().zip(predicted.iter()) {
        for (exp_val, pred_val) in exp_row.iter().zip(pred_row.iter()) {
            let error = exp_val - pred_val;
            mse += error * error;
        }
    }

    mse /= (expected.len() * expected[0].len()) as f64;
    mse
}

fn are_equal<T: PartialEq>(a: &Vec<Vec<T>>, b: &Vec<Vec<T>>) -> bool {
    if a.len() != b.len() {
        return false;
    }

    for (row_a, row_b) in a.iter().zip(b.iter()) {
        if row_a.len() != row_b.len() {
            return false;
        }

        for (elem_a, elem_b) in row_a.iter().zip(row_b.iter()) {
            if elem_a != elem_b {
                return false;
            }
        }
    }

    true
}

impl NeuralNetwork {
    pub fn new() -> NeuralNetwork {
        NeuralNetwork { layers: Vec::new(), loss: Vec::new() }
    }

    pub fn add<F>(mut self, layer: Box<dyn Layer>) -> NeuralNetwork {
        self.layers.push(layer);
        self
    }

    pub fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let mut output = input.clone();
        for layer in &mut self.layers {
            output = layer.forward(&output);
        }
        output
    }

    pub fn fit(
        &mut self,
        x_train: &Vec<Vec<f64>>,
        y_train: &Vec<Vec<f64>>,
        epochs: usize,
    ) {

        for i in 0..epochs {
            let mut misclassifications = 0;
            println!("epochs: {}", i);
            // Go through all the images
            for j in 0..x_train.len() {
                let mut errors: Vec<Vec<f64>> = Vec::new();
                
                // Forward propagate the values
                let mut output = Vec::new();
                output.push(x_train[j].clone());
                for layer in &mut self.layers {
                    output = layer.forward(&output);
                }

                // Calculate the error at the end
                for k in 0..output.len() {
                    errors[0].push(output[0][k] - y_train[j][k]);
                }

                // Store the loss
                // let metrics = Metrics::new(&output, &y_train[j]);
                let mut actual = Vec::new();
                actual.push(y_train[j].clone());
                self.loss.push(mean_square_error(&output, &actual));
                let predicted = apply_activation(&actual, 1, y_train[0].len(), Activation::BinaryActivation);
                
                if !are_equal(&predicted, &actual) {
                    misclassifications += 1;
                }

                if j % 500 == 0 {
                    println!("images processed: {}", j);
                    println!("last loss: {:?}", self.loss.last());
                }

                // Backpropagate the errors
                for layer in self.layers.iter_mut().rev() {
                    errors = layer.backward(&errors);
                }
            }

            println!("misclassifications: {}", misclassifications);
        }
    }
}
