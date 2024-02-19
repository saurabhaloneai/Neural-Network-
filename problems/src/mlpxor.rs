extern crate rand;

use rand::Rng;

struct MLP {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    weights_ih: Vec<Vec<f64>>,
    weights_ho: Vec<Vec<f64>>,
    bias_h: Vec<f64>,
    bias_o: f64,
    learning_rate: f64,
}

impl MLP {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> MLP {
        let mut rng = rand::thread_rng();

        let weights_ih: Vec<Vec<f64>> = (0..hidden_size)
            .map(|_| (0..input_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let weights_ho: Vec<Vec<f64>> = (0..output_size)
            .map(|_| (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect())
            .collect();

        let bias_h: Vec<f64> = (0..hidden_size).map(|_| rng.gen_range(-1.0..1.0)).collect();
        let bias_o = rng.gen_range(-1.0..1.0);
        let learning_rate = 0.1;

        MLP {
            input_size,
            hidden_size,
            output_size,
            weights_ih,
            weights_ho,
            bias_h,
            bias_o,
            learning_rate,
        }
    }

    fn activate(&self, sum: f64) -> f64 {
        1.0 / (1.0 + (-sum).exp())
    }

    fn feedforward(&self, inputs: &[f64]) -> Vec<f64> {
        // Calculate hidden layer outputs
        let mut hidden_outputs = vec![0.0; self.hidden_size];
        for i in 0..self.hidden_size {
            let mut sum = 0.0;
            for j in 0..self.input_size {
                sum += inputs[j] * self.weights_ih[i][j];
            }
            sum += self.bias_h[i];
            hidden_outputs[i] = self.activate(sum);
        }

        // Calculate final outputs
        let mut final_outputs = vec![0.0; self.output_size];
        for i in 0..self.output_size {
            let mut sum = 0.0;
            for j in 0..self.hidden_size {
                sum += hidden_outputs[j] * self.weights_ho[i][j];
            }
            sum += self.bias_o;
            final_outputs[i] = self.activate(sum);
        }

        final_outputs
    }

    fn train(&mut self, inputs: &[f64], targets: &[f64]) {
        // Feedforward
        let hidden_outputs = self.feedforward(inputs);

        // Calculate output layer errors
        let output_errors: Vec<f64> = (0..self.output_size)
            .map(|i| targets[i] - hidden_outputs[i])
            .collect();

        // Calculate output layer gradients
        let output_gradients: Vec<f64> = (0..self.output_size)
            .map(|i| hidden_outputs[i] * (1.0 - hidden_outputs[i]) * output_errors[i])
            .collect();

        // Update output layer weights and biases
        for i in 0..self.output_size {
            for j in 0..self.hidden_size {
                self.weights_ho[i][j] += self.learning_rate * output_gradients[i] * hidden_outputs[j];
            }
            self.bias_o += self.learning_rate * output_gradients[i];
        }

        // Calculate hidden layer errors
        let hidden_errors: Vec<f64> = (0..self.hidden_size)
            .map(|i| {
                let mut sum = 0.0;
                for j in 0..self.output_size {
                    sum += self.weights_ho[j][i] * output_errors[j];
                }
                hidden_outputs[i] * (1.0 - hidden_outputs[i]) * sum
            })
            .collect();

        // Calculate hidden layer gradients
        let hidden_gradients: Vec<f64> = (0..self.hidden_size)
            .map(|i| inputs[i] * (1.0 - inputs[i]) * hidden_errors[i])
            .collect();

        // Update hidden layer weights and biases
        for i in 0..self.hidden_size {
            for j in 0..self.input_size {
                self.weights_ih[i][j] += self.learning_rate * hidden_gradients[i] * inputs[j];
            }
            self.bias_h[i] += self.learning_rate * hidden_gradients[i];
        }
    }
     fn predict(&self, inputs: &[f64]) -> Vec<f64> {
        self.feedforward(inputs)
    }
}

fn main() {
    let mut mlp = MLP::new(2, 4, 1);

    let training_data = [
        ([0.0, 0.0], [0.0]),
        ([0.0, 1.0], [1.0]),
        ([1.0, 0.0], [1.0]),
        ([1.0, 1.0], [0.0]),
    ];

    // Training the MLP
    for _ in 0..10000 {
        for &(inputs, targets) in &training_data {
            mlp.train(&inputs, &targets);
        }
    }

    // Testing the MLP
    for &(inputs, _) in &training_data {
        let prediction = mlp.predict(&inputs)[0];
        println!("Input: {:?}, Output: {:.4}", inputs, prediction);
    }
}
 
    
    


