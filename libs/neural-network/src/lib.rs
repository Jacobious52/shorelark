#![feature(array_windows)]

use rand::{Rng, RngCore};

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn random(rng: &mut dyn RngCore, layers: &[LayerTopology]) -> Self {
        assert!(layers.len() > 1);

        let layers = layers
            .array_windows()
            .map(|[first, second]| Layer::random(rng, first.neurons, second.neurons))
            .collect();

        Self { layers }
    }

    pub fn propagate(&self, inputs: Vec<f32>) -> Vec<f32> {
        self.layers
            .iter()
            .fold(inputs, |inputs, layer| layer.propagate(&inputs))
    }
}

pub struct LayerTopology {
    pub neurons: usize,
}

struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn random(rng: &mut dyn RngCore, input_neurons: usize, output_neurons: usize) -> Self {
        let neurons = (0..output_neurons)
            .map(|_| Neuron::random(rng, input_neurons))
            .collect();

        Self { neurons }
    }

    fn propagate(&self, inputs: &[f32]) -> Vec<f32> {
        self.neurons
            .iter()
            .map(|neuron| neuron.propagate(&inputs))
            .collect()
    }
}

#[derive(Debug, Clone)]
struct Neuron {
    bias: f32,
    weights: Vec<f32>,
}

impl Neuron {
    pub fn random(rng: &mut dyn RngCore, output_size: usize) -> Self {
        let bias = rng.gen_range(-1.0..=1.0);

        let weights = (0..output_size)
            .map(|_| rng.gen_range(-1.0..=1.0))
            .collect();

        Self { bias, weights }
    }

    fn propagate(&self, inputs: &[f32]) -> f32 {
        assert_eq!(inputs.len(), self.weights.len());

        let output = inputs
            .iter()
            .zip(&self.weights)
            .map(|(input, weight)| input * weight)
            .sum::<f32>();

        (self.bias + output).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    mod random {
        use super::*;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        #[test]
        fn test_neuron() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let neuron = Neuron::random(&mut rng, 4);

            approx::assert_relative_eq!(neuron.bias, -0.6255188);

            approx::assert_relative_eq!(
                neuron.weights.as_slice(),
                [0.67383957, 0.8181262, 0.26284897, 0.5238807].as_ref(),
            );
        }

        #[test]
        fn test_layer() {
            let mut rng = ChaCha8Rng::from_seed(Default::default());
            let layer = Layer::random(&mut rng, 3, 2);

            let actual_biases: Vec<_> = layer.neurons.iter().map(|neuron| neuron.bias).collect();
            let expected_biases = vec![-0.6255188, 0.5238807];

            let actual_weights: Vec<_> = layer
                .neurons
                .iter()
                .map(|neuron| neuron.weights.as_slice())
                .collect();
            let expected_weights: Vec<&[f32]> = vec![
                &[0.67383957, 0.8181262, 0.26284897],
                &[-0.53516835, 0.069369674, -0.7648182],
            ];

            approx::assert_relative_eq!(actual_biases.as_slice(), expected_biases.as_slice());
            approx::assert_relative_eq!(actual_weights.as_slice(), expected_weights.as_slice());
        }
    }

    mod propagate {
        use super::*;

        #[test]
        fn test_neuron() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            assert_relative_eq!(neuron.propagate(&[-10.0, -10.0]), 0.0,);

            approx::assert_relative_eq!(
                neuron.propagate(&[0.5, 1.0]),
                (-0.3 * 0.5) + (0.8 * 1.0) + 0.5,
            );
        }

        #[test]
        fn test_layer_one_neuron() {
            let neuron = Neuron {
                bias: 0.5,
                weights: vec![-0.3, 0.8],
            };

            let layer = Layer {
                neurons: vec![neuron],
            };

            let output = layer.propagate(&[-10.0, -10.0]);
            assert_relative_eq!(output.as_slice(), [0.0].as_ref(),);
        }

        #[test]
        fn test_layer_two_neuron() {
            let neurons = (
                Neuron {
                    bias: 0.0,
                    weights: vec![0.1, 0.2, 0.3],
                },
                Neuron {
                    bias: 0.0,
                    weights: vec![0.4, 0.5, 0.6],
                },
            );

            let layer = Layer {
                neurons: vec![neurons.0.clone(), neurons.1.clone()],
            };

            let inputs = &[-0.5, 0.0, 0.5];

            let actual = layer.propagate(inputs);
            let expected = vec![neurons.0.propagate(inputs), neurons.1.propagate(inputs)];
            assert_relative_eq!(actual.as_slice(), expected.as_slice(),);
        }
    }
}
