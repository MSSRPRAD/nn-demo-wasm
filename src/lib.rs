pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}
#[no_mangle]
fn matadd(
    matrix_a: &Vec<f64>,
    matrix_b: &Vec<f64>,
    len: usize,
    ) -> Vec<f64> {
    let mut result = Vec::new();
    for i in 0..len {
        result.push(matrix_a[i]+matrix_b[i]);
    }
    result
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
    fn forward(&self, input: &Vec<f64>) -> Vec<f64>;
    fn backward(&self, output_errors: &Vec<f64>) -> Vec<f64>;
}
#[no_mangle]
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
fn transpose(vector: Vec<Vec<f64>>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
   let mut result: Vec<Vec<f64>> = vec![vec![0.0; rows]; cols];
   for i in 0..cols {
       for j in 0..rows {
           result[i][j] = vector[j][i]
       }
   }
   result
}
#[no_mangle]
fn apply_activation(vector: &[f64], len: usize, activation: Activation) -> Vec<f64> {
    let mut result = vec![0.0; len];

    match activation {
        Activation::Tanh => {
            for (i, &value) in vector.iter().enumerate() {
                result[i] = value.tanh();
            }
        }
        Activation::Sigmoid => {
            for (i, &value) in vector.iter().enumerate() {
                result[i] = 1.0 / (1.0 + (-value).exp());
            }
        }
        Activation::Relu => {
            for (i, &value) in vector.iter().enumerate() {
                result[i] = value.max(0.0);
            }
        }
        Activation::Softmax => {
            let max_value = vector.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = vector.iter().map(|&x| (x - max_value).exp()).sum();

            for (i, &value) in vector.iter().enumerate() {
                result[i] = (value - max_value).exp() / exp_sum;
            }
        }
    }

    result
}

#[no_mangle]
pub struct DenseLayer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    inputs: Vec<f64>,
    outputs: Vec<f64>,
    activations: Vec<f64>,
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
            inputs: vec![0.0; input_size],
            outputs: vec![0.0; output_size],
            activations: vec![0.0; output_size],
            activation: Activation::from_int(activation).unwrap(),
            weights: vec![vec![0.0; output_size]; input_size],
            biases: vec![0.0; output_size],
        }
    }
}
// input: 1x10
// output: 1x2
// weights: 10x2
impl Layer for DenseLayer {
    fn forward(&self, input: &Vec<f64>) -> Vec<f64> {
        self.activations = matadd(&matmul(&self.inputs, 1, self.input_size, &self.weights, self.input_size, self.output_size).unwrap()[0], &self.biases, self.input_size);
        self.outputs.clone()
    }
    fn backward(&self, output_errors: &Vec<f64>) -> Vec<f64> {
        self.inputs.clone()
    }
}
