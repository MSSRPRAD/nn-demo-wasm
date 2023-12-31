// lib.rs
pub mod nn {
    #[no_mangle]
    pub fn multiply_elements(matrix: &Vec<Vec<f64>>, factor: f64) -> Vec<Vec<f64>> {
        let mut result = matrix.clone(); // Create a clone of the matrix

        for row in result.iter_mut() {
            for element in row.iter_mut() {
                *element *= factor;
            }
        }

        result // Return the modified matrix
    }
    #[no_mangle]
    pub fn mul(matrix_a: &Vec<Vec<f64>>, num: f64) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(matrix_a.len());
        for row in matrix_a {
            let mut new_row = Vec::with_capacity(row.len());
            for &value in row {
                new_row.push(value * num);
            }
            result.push(new_row);
        }
        result
    }

    #[no_mangle]
    pub fn matadd(
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
    pub fn matsub(
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
    pub fn matsub_lr(
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
    pub fn pairwisemul(
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
    pub fn matmul(
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
        fn print(&self) -> ();
    }

    #[derive(Clone, Copy, Debug)]
    pub enum Activation {
        Tanh = 0,
        Sigmoid = 1,
        Relu = 2,
        Softmax = 3,
        BinaryActivation = 4,
        Identity = 5,
    }

    impl Activation {
        fn from_int(value: usize) -> Option<Activation> {
            match value {
                0 => Some(Activation::Tanh),
                1 => Some(Activation::Sigmoid),
                2 => Some(Activation::Relu),
                3 => Some(Activation::Softmax),
                4 => Some(Activation::BinaryActivation),
                5 => Some(Activation::Identity),
                _ => None,
            }
        }
    }
    #[no_mangle]
    pub fn transpose(matrix: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
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
    pub fn predict(vector: &Vec<Vec<f64>>, rows: usize, cols: usize) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; cols]; rows];

        for i in 0..rows {
            let max_index = vector[i]
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(index, _)| index);
            if let Some(index) = max_index {
                result[i][index] = 1.0;
            }
        }

        result
    }
    #[no_mangle]
    pub fn apply_activation(
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
                    Activation::Identity => {
                        result[i][j] = vector[i][j];
                    }
                }
            }
        }

        result
    }

    #[no_mangle]
    pub fn activation_derivative(
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
                    Activation::Identity => {
                        result[i][j] = 0.0;
                    }
                    Activation::Softmax | _ => {
                        panic!("Derivative is either not defined or implemented!");
                    }
                }
            }
        }

        result
    }

    #[derive(Clone, Debug)]
    pub struct DenseLayer {
        batch_length: usize,
        batch_count: usize,
        output_errors_avg: Vec<Vec<f64>>,
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
    use rand::prelude::*;
    impl DenseLayer {
        // weights is an array of input_size rows and output_size columns
        // i-th row j-th column is the weight connecting i-th input neuron to j-th output neuron
        pub fn new(
            input_size: usize,
            output_size: usize,
            lr: f64,
            activation: usize,
            batch_length: usize,
        ) -> DenseLayer {
            DenseLayer {
                input_size,
                output_size,
                lr,
                batch_length,
                batch_count: 0,
                output_errors_avg: vec![vec![0.0; output_size]],
                input_errors: vec![vec![0.0; input_size]],
                output_errors: vec![vec![0.0; output_size]],
                inputs: vec![vec![0.0; input_size]],
                outputs: vec![vec![0.0; output_size]],
                activations: vec![vec![0.0; output_size]],
                activation: Activation::from_int(activation).unwrap(),
                weights: vec![vec![thread_rng().gen(); output_size]; input_size],
                biases: vec![vec![thread_rng().gen(); output_size]],
            }
        }        
    }
    // input: 1x10
    // output: 1x2
    // weights: 10x2

    impl Layer for DenseLayer {
        fn print(&self) -> () {
            println!("Dense Layer Structure:");
            println!("({:?}, {:?})", self.input_size, self.output_size);
            println!("Weights:");
            println!("{:?}", self.weights);
            println!("Inputs:");
            println!("{:?}", self.inputs);
            println!("Input Errors:");
            println!("{:?}", self.input_errors);
        }
        // [1,IS]x[IS,OS] = [1,OS]
        fn forward(&mut self, input: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
            self.inputs = input.clone();
            // println!("starting forward!");
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

            self.outputs =
                apply_activation(&self.activations, 1, self.output_size, self.activation);
            // println!("self.outputs: {:?}", self.outputs);
            self.outputs.clone()
        }
        // self.inputs: 1x256
        // self.weights:
        fn backward(&mut self, output_errors: &Vec<Vec<f64>>) -> Vec<Vec<f64>> {
            println!("starting backward!");
            println!("output_errors: {:?}", output_errors);
            self.output_errors = pairwisemul(&activation_derivative(&self.activations, 1, self.output_size, self.activation), &output_errors, 1, self.output_size);
            println!("output_errors (after ac_der): {:?}", self.output_errors);
            self.input_errors = matmul(&self.output_errors, 1, self.output_size, &transpose(&self.weights), self.output_size, self.input_size).unwrap();
            println!("self.input_errors: {:?}", self.input_errors);
            self.batch_count += 1;
            // println!("adding to self.output_errors_avg...");
            // println!("output_errors: {:?}", self.output_errors);
            // println!("output_errors_avg: {:?}", self.output_errors_avg);
            self.output_errors_avg = matadd(
                &self.output_errors_avg,
                &self.output_errors,
                1,
                self.output_size,
            );
            println!("self.output_errors_avg (self.batch_count = {:?}) : {:?}", self.batch_count, self.output_errors_avg);
            if self.batch_count == self.batch_length {
                println!("before updating, weights: {:?}", self.weights);
                println!("self.output_errors_avg: {:?}", self.output_errors_avg);
                self.update_weights();
                println!("after updating, weights: {:?}", self.weights);
            }

            // println!("weight errors: {:?}", weight_errors);
            // println!("weights: {:?}", self.weights);
            // println!("difference in weights: {:?}", matsub(&initial_weights, &self.weights, self.input_size, self.output_size));
            println!("return from backward_propogation: {:?}", self.input_errors);
            self.input_errors.clone()
        }
    }
    impl DenseLayer {
        pub fn update_weights(&mut self) -> () {
            println!("updating weights...");
            self.output_errors_avg =
                multiply_elements(&self.output_errors_avg, 1.0 / (self.batch_length as f64));
            self.batch_count = 0;
            println!("self.output_errors_avg: {:?}", self.output_errors_avg);
            println!("self.inputs: {:?}", self.inputs);
            let weight_errors = matmul(
                &transpose(&self.inputs),
                self.input_size,
                1,
                &self.output_errors_avg,
                1,
                self.output_size,
            )
            .unwrap();
            println!("weight errors: {:?}", weight_errors);
            // let initial_weights = self.weights.clone();
            self.weights = matsub_lr(
                &self.weights,
                &weight_errors,
                self.input_size,
                self.output_size,
                self.lr,
            );
            // println!("weights: {:?}", self.weights);
            // println!("difference in weights: {:?}", matsub(&initial_weights, &self.weights, self.input_size, self.output_size));
            self.biases = matsub_lr(
                &self.biases,
                &self.output_errors_avg,
                1,
                self.output_size,
                self.lr,
            );
            self.output_errors_avg = vec![vec![0.0; self.output_size]];
        }
    }
    pub struct NeuralNetwork {
        pub layers: Vec<Box<dyn Layer>>,
        pub loss: Vec<f64>,
        pub use_binary: bool,
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
        pub fn new(use_binary: bool) -> NeuralNetwork {
            NeuralNetwork {
                layers: Vec::new(),
                loss: Vec::new(),
                use_binary,
            }
        }
        pub fn add<F>(&mut self, layer: Box<dyn Layer>) -> &mut Self {
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

        pub fn fit<F>(
            &mut self,
            x_train: &Vec<Vec<f64>>,
            y_train: &Vec<Vec<f64>>,
            epochs: usize,
            interval: usize,
            callback: F,
        ) where
            F: Fn(&NeuralNetwork),
        {
            for i in 0..epochs {
                let mut misclassifications = 0;
                // println!("epochs: {}", i);
                // Go through all the tuples
                for j in 0..x_train.len() {
                    let actual = vec![y_train[j].clone()];
                    let mut errors: Vec<Vec<f64>> = vec![vec![0.0; actual[0].len()]];

                    // println!("errors: {:?}", errors);
                    // Forward propagate the values
                    let mut output = Vec::new();

                    output.push(x_train[j].clone());
                    // println!("output initially: {:?}", output);
                    for layer in &mut self.layers {
                        output = layer.forward(&output);
                    }

                    // Calculate the error at the end
                    errors = matsub(&output, &actual, actual.len(), actual[0].len());

                    // println!("errors: {:?}", errors);
                    // Store the loss

                    self.loss.push(mean_square_error(&output, &actual));
                    let predicted;
                    if !self.use_binary {
                        predicted = predict(&output, actual.len(), actual[0].len());
                    } else {
                        predicted = apply_activation(
                            &output,
                            actual.len(),
                            actual[0].len(),
                            Activation::BinaryActivation,
                        );
                    }
                    // println!("output: {:?}", output);
                    // println!("predicted: {:?}", predicted);
                    // println!("actual: {:?}", actual);
                    // println!("self.weights: {:?}", self.weights);
                    if !are_equal(&predicted, &actual) {
                        // println!("errors: {:?}", errors);
                        misclassifications += 1;
                    }

                    if (j + 1) % interval == 0 {
                        callback(self);
                        println!("tuple: {:?}", x_train[j].clone());
                        println!("output: {:?}", output);
                        println!("predicted: {:?}", predicted);
                        println!("actual: {:?}", actual);

                        println!("images processed: {}", j);
                        println!("loss: {:?}", self.loss.last());
                    }
                    // Backpropagate the errors
                    println!("errors: {:?}", errors);
                    for layer in self.layers.iter_mut().rev() {
                        // println!("errors: {:?}", errors);
                        errors = layer.backward(&errors);
                    }
                    // println!("first input errors: {:?}", errors);
                }

                println!("epoch: {} | misclassifications: {}", i, misclassifications);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::nn::*;

    #[test]
    fn test_matmul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![3.0, 2.0]];
        let b = vec![vec![1.0, 2.0, 2.0, 4.0], vec![2.0, 3.0, 1.0, 1.0]];
        assert_eq!(
            matmul(&a, 3, 2, &b, 2, 4).unwrap(),
            [
                [5.0, 8.0, 4.0, 6.0],
                [11.0, 18.0, 10.0, 16.0],
                [7.0, 12.0, 8.0, 14.0]
            ]
        )
    }

    #[test]
    fn test_matadd_matsub() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![3.0, 2.0]];
        let b = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![3.0, 2.0]];
        assert_eq!(matadd(&a, &b, 3, 2), [[2.0, 4.0], [6.0, 8.0], [6.0, 4.0]]);

        assert_eq!(matsub(&a, &b, 3, 2), [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    }

    #[test]
    fn test_transpose() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![3.0, 2.0]];

        assert_eq!(transpose(&a), [[1.0, 3.0, 3.0], [2.0, 4.0, 2.0]])
    }

    #[test]
    fn test_pairwisemul() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![3.0, 2.0]];
        let b = vec![vec![2.0, 2.0], vec![3.0, 3.0], vec![1.0, 1.0]];

        assert_eq!(
            pairwisemul(&a, &b, 3, 2),
            [[2.0, 4.0], [9.0, 12.0], [3.0, 2.0]]
        )
    }
}
