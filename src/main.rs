//https://www.infoworld.com/article/3685569/how-to-build-a-neural-network-in-java.html
mod network;
pub mod neuron;

use crate::network::Learnable;

fn main() {
    println!("Welcome to the neural net example!");
    println!("we will train a neural net to preduct if a new input is a man or a woman based on their weight and height.\n");

    // naive prediction:
    // let prediction = Network::default().predict(115 as f64, 66 as f64);
    // println!("prediction:{}", prediction);

    //weight, height
    let data:Vec<Vec<i32>> = vec![vec![115, 66], vec![175, 78], vec![205, 72], vec![120, 67]];

    let answers: Vec<f64> = vec![1.0,0.0,0.0,1.0];  

    let mut network = network::Network::default();
    network.train(data, answers);

    test(&network);
}

fn test(network: &network::Network) {
    println!("\n  male: 167, 73: {}", network.predict(167.0, 73.0));
    println!("female: 105, 67: {}", network.predict(105.0, 67.0)); 
    println!("female: 120, 72: {}", network.predict(120.0, 72.0)); 
    println!("  male: 143, 67: {}", network.predict(143.0, 67.0));
    
    //print outlier
    println!("  male', 130, 66: {} <-- deliberate outlier", network.predict(130.0, 66.0));
}


