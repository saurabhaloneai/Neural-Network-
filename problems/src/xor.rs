mod mlpxor;

extern crate rand;

use rand::Rng;

struct Perceptron {
    weights: [f64; 2],
    bias: f64,
    learning_rate: f64,
}

impl Perceptron {
    fn new() -> Perceptron {
        let mut rng = rand::thread_rng();
        let weights = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
        let bias = rng.gen_range(-1.0..1.0);
        let learning_rate = 0.1;

        Perceptron {
            weights,
            bias,
            learning_rate,
        }
    }

    fn activate(&self, sum: f64) -> i32 {
        if sum > 0.0 {
            1
        } else {
            0
        }
    }

    fn train(&mut self, inputs: &[f64; 2], target: i32) {
        let mut sum = 0.0;
        for i in 0..inputs.len() {
            sum += inputs[i] * self.weights[i];
        }
        sum += self.bias;

        let output = self.activate(sum);

        let error = target - output;
        for i in 0..self.weights.len() {
            self.weights[i] += self.learning_rate * error as f64 * inputs[i];
        }
        self.bias += self.learning_rate * error as f64;
    }

    fn predict(&self, inputs: &[f64; 2]) -> i32 {
        let mut sum = 0.0;
        for i in 0..inputs.len() {
            sum += inputs[i] * self.weights[i];
        }
        sum += self.bias;

        self.activate(sum)
    }
}

fn main() {
    let mut perceptron = Perceptron::new();

    let training_data = [
        ([0.0, 0.0], 0),
        ([0.0, 1.0], 1),
        ([1.0, 0.0], 1),
        ([1.0, 1.0], 0),
    ];

    // training the perceptron
    for _ in 0..10000 {
        for &(inputs, target) in &training_data {
            perceptron.train(&inputs, target);
        }
    }

    // testing the perceptron
    for &(inputs, _) in &training_data {
        let prediction = perceptron.predict(&inputs);
        println!("Input: {:?}, Output: {}", inputs, prediction);
    }
}
