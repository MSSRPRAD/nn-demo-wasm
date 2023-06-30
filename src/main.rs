use nn_demo_wasm::nn::*;

// fn duplicate_dataset(dataset: &Vec<Vec<f64>>, num_rows: usize) -> Vec<Vec<f64>> {
//     let mut duplicated_dataset = vec![];
//
//     for _ in 0..num_rows {
//         duplicated_dataset.extend_from_slice(dataset);
//     }
//
//     duplicated_dataset
// }

use csv;

use std::error::Error;
use std::fs::File;
use std::str::FromStr;

fn convert_vector(y_train: Vec<f64>) -> Vec<Vec<f64>> {
    let mut converted_vector: Vec<Vec<f64>> = Vec::new();

    for value in y_train {
        // println!("{:?}", value);
        let mut converted: Vec<f64> = vec![0.0; 10];

        converted[value as usize] = 1.0;

        converted_vector.push(converted);
    }

    converted_vector
}

fn read_csv_file(path: &str) -> Result<(Vec<Vec<f64>>, Vec<Vec<f64>>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = csv::Reader::from_reader(file);

    let mut first_column: Vec<f64> = Vec::new();
    let mut rest_columns: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;

        let mut row: Vec<f64> = Vec::new();

        for (i, value) in record.iter().enumerate() {
            if i == 0 {
                let float_value = value.parse::<f64>()?;
                first_column.push(float_value);
            } else {
                let float_value = value.parse::<f64>()?;
                row.push(float_value);
            }
        }

        rest_columns.push(row);
    }
    
    Ok((convert_vector(first_column), rest_columns))
}

fn main() {
    let (y_train, x_train) = read_csv_file("./mnist/mnist_train.csv").unwrap();
    println!("x_train[0]: {:?}", x_train[0]);
    println!("y_train[0]: {:?}", y_train[0]);
    println!("x_train[1]: {:?}", x_train[1]);
    println!("y_train[1]: {:?}", y_train[1]);

    let mut model = NeuralNetwork::new();
    let layer: Box<dyn Layer> = Box::new(DenseLayer::new(784, 50, 0.001, 2));
    model.add::<Box<dyn Layer>>(layer);
    let layer: Box<dyn Layer> = Box::new(DenseLayer::new(50, 10, 0.001, 5));
    model.add::<Box<dyn Layer>>(layer);

    // Define the callback function to print the weights
    let print_weights_callback = |model: &NeuralNetwork| {
        for layer in &model.layers {
            layer.print();
        }
    };

    model.fit(&x_train, &y_train, 1, 2000, print_weights_callback);

    println!("Final Weights:");
    println!("{:?}\n{:?}", model.layers[0].print(), model.layers[1].print());
}
