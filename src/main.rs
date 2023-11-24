use neuron::{Neuron, BackPropogatable};
use crate::neuron::Compute;
//https://www.infoworld.com/article/3685569/how-to-build-a-neural-network-in-java.html
mod neuron;

fn main() {
    println!("Welcome to the neural net example!");
    println!("we will train a neural net to preduct if a new input is a man or a woman based on their weight and height.\n");

    // naive prediction:
    // let prediction = Network::default().predict(115 as f64, 66 as f64);
    // println!("prediction:{}", prediction);

    //weight, height
    let data:Vec<Vec<i32>> = vec![vec![115, 66], vec![175, 78], vec![205, 72], vec![120, 67]];

    let answers: Vec<f64> = vec![1.0,0.0,0.0,1.0];  

    let mut network = Network::default();
    network.train(data, answers);

    test(&network);
}

fn test(network: &Network) {
    println!("\n  male: 167, 73: {}", network.predict(167.0, 73.0));
    println!("female: 105, 67: {}", network.predict(105.0, 67.0)); 
    println!("female: 120, 72: {}", network.predict(120.0, 72.0)); 
    println!("  male: 143, 67: {}", network.predict(143.0, 67.0));
    
    //print outlier
    println!("  male', 130, 66: {} <-- deliberate outlier", network.predict(130.0, 66.0));
}

//although this is a 1d list, we connect them during usage
struct Network{
    neurons: Vec<neuron::Neuron>,

}

// The number of hidden neurons should be 2/3 the size of the input layer, plus the size of the output layer. 
// The number of hidden neurons should be less than twice the size of the input layer.
impl Default for Network {
    fn default() -> Self {
        let neurons = vec![
            Neuron::default(), Neuron::default(), Neuron::default(), //input nodes
            Neuron::default(), Neuron::default(), //hidden nodes
            Neuron::default()]; //output nodes
        Network { neurons }
    }
}
trait Learnable {
    fn predict(&self, input1: f64, input2: f64) -> f64;
    fn train(&mut self, data: Vec<Vec<i32>>, answers: Vec<f64>);
}

impl Learnable for Network {
    fn predict(&self, input1: f64, input2: f64) -> f64 {
        self.neurons[5].compute(
            self.neurons[4].compute(
                self.neurons[2].compute(input1, input2), 
                self.neurons[1].compute(input1, input2)
            ),
            self.neurons[3].compute(
                self.neurons[1].compute(input1, input2), 
                self.neurons[0].compute(input1, input2)
            )
        )
    }

    fn train(&mut self, data: Vec<Vec<i32>>, answers: Vec<f64>) {
        let mut best_epoch_loss =  -1.0;
        for epoch in 0..1000 {
            // adapt neuron
            self.neurons[epoch % 6].mutate();

            let mut predictions:Vec<f64> = Vec::new();

            for item in data.iter().take(data.len() -1){
                predictions.push(self.predict(item[0] as f64, item[1] as f64));
            }

            let this_epoch_loss = mean_square_loss(&answers, predictions);

            if best_epoch_loss == -1.0 {
                best_epoch_loss = this_epoch_loss;
                self.neurons[epoch % 6].remember();
            } else {
                if this_epoch_loss < best_epoch_loss {
                best_epoch_loss = this_epoch_loss;
                self.neurons[epoch % 6].remember();
            } else {
                self.neurons[epoch % 6].forget();
            }

            //log every 10th training epoch
            if epoch % 10 == 0{
                println!("Epoch: {} | bestEpochLoss: {} | thisEpochLoss: {}", epoch, best_epoch_loss, this_epoch_loss);
            } 
        }
    }
}
}

fn mean_square_loss(correct_answers: &Vec<f64>, predicted_answers: Vec<f64>) -> f64{
    let mut sum_square = 0.0;
    for i in 0..correct_answers.len() -1 {
      let error = correct_answers[i] - predicted_answers[i];
      sum_square += error * error;
    }
    sum_square / (correct_answers.len() as f64)
}

