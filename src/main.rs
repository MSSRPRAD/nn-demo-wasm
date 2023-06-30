use nn_demo_wasm::nn::*;
fn main() {
    let x_train: Vec<Vec<f64>> = vec![
        vec![0.00, 0.00],
        vec![0.00, 1.00],
        vec![1.00, 0.00],
        vec![1.00, 1.00],
    ];
    let y_train: Vec<Vec<f64>> = vec![vec![0.00], vec![1.00], vec![1.00], vec![0.00]];
   
    let mut model = NeuralNetwork::new();
    let layer: Box<dyn Layer> = Box::new(DenseLayer::new(2, 50, 0.01, 1));
    model.add::<Box<dyn Layer>>(layer);
    let layer: Box<dyn Layer> = Box::new(DenseLayer::new(50, 1, 0.01, 1));
    model.add::<Box<dyn Layer>>(layer);

    // Define the callback function to print the weights
    let print_weights_callback = |model: &NeuralNetwork| {
        for layer in &model.layers {
            layer.print();
        }
    };

    model.fit(&x_train, &y_train, 100000, 1, print_weights_callback);
}
