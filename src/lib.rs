#[no_mangle]
fn matadd(
    matrix_a: &Vec<Vec<f64>>,
    matrix_b: &Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
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
    rows: usize,
    cols: usize,
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
    rows: usize,
    cols: usize,
    lr: f64,
) -> Vec<Vec<f64>> {
    matrix_a
        .iter()
        .zip(matrix_b.iter())
        .map(|(row_a, row_b)| {
            row_a
                .iter()
                .zip(row_b.iter())
                .map(|(&a, &b)| a - lr*b)
                .collect::<Vec<f64>>()
        })
        .collect()
}

#[no_mangle]
fn pairwisemul(
    matrix_a: &Vec<Vec<f64>>,
    matrix_b: &Vec<Vec<f64>>,
    rows: usize,
    cols: usize,
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

#[no_mangle]
pub trait Layer {
    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
    fn backward(&mut self, output_errors: &Vec<Vec<f64>>) -> Vec<Vec<f64>>;
}
#[no_mangle]
#[derive(Clone, Copy, Debug)]
enum Activation {
    Tanh = 0,
    Sigmoid = 1,
    Relu = 2,
    Softmax = 3,
}

impl Activation {
    fn from_int(value: usize) -> Option<Activation> {
        match value {
            0 => Some(Activation::Tanh),
            1 => Some(Activation::Sigmoid),
            2 => Some(Activation::Relu),
            3 => Some(Activation::Softmax),
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
                Activation::Softmax => {
                    panic!("Derivative of softmax function is not defined.");
                }
            }
        }
    }

    result
}

#[no_mangle]
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
    fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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
        // self.input_errors = ac.relu_der(self.inputs)*(np.dot(self.weights, output_errors.T)).T
        // self.input_errors = activation_derivative(self.inputs, rows, cols, activation)
        //  # Calculate the error in weights for this layer
    // weight_errors = np.dot(self.inputs.T, output_errors)
    // # Update the weights
    // self.weights -= lr*weight_errors
    // # Update the bias
    // self.bias -= lr*output_errors
    // return self.input_errors
        let weight_errors = matmul(&transpose(&self.inputs), 1, self.input_size, &output_errors, 1, self.output_size).unwrap();
        self.weights = matsub_lr(&self.weights,&weight_errors, self.input_size, self.output_size, self.lr);
        self.biases = matsub_lr(&self.biases, &output_errors, 1, self.output_size, self.lr); 
        self.input_errors.clone()
    }
}
