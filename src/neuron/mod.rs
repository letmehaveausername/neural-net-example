use rand::Rng;

pub struct Neuron {
    bias: f64,
    weight1: f64,
    weight2: f64,
    old_bias: f64,
    old_weight1: f64,
    old_weight2: f64,
}

impl Default for Neuron {
    fn default() -> Self {
        let mut rng = rand::thread_rng();
        let init_bias = rng.gen_range(-1.0..1.0);
        let init_weight1 = rng.gen_range(-1.0..1.0);
        let init_weight2 = rng.gen_range(-1.0..1.0);

        Neuron { bias: init_bias, 
            weight1: init_weight1, 
            weight2: init_weight2,
            old_bias: init_bias,
            old_weight1: init_weight1,
            old_weight2: init_weight2,
         }
    }
}

pub trait Compute {
    fn compute(&self, input1: f64, input2: f64) -> f64;
}

pub trait BackPropogatable {
    fn mutate(&mut self);
    fn forget(&mut self);
    fn remember(&mut self);
}

impl Compute for Neuron {
    fn compute(&self, input1: f64, input2: f64) -> f64 {
        let pre_activation = 
        (self.weight1 * input1) + 
        (self.weight2 * input2) + 
        self.bias;

        sigmoid(pre_activation)
    }
}

impl BackPropogatable for Neuron {
    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let property_to_change = rng.gen_range(0..3);
        let change_factor = rng.gen_range(-1.0..1.0);

        if property_to_change == 0 { 
            self.bias += change_factor; 
        } else if property_to_change == 1 { 
            self.weight1 += change_factor; 
        } else { 
            self.weight2 += change_factor; 
        };
    }
    
    fn forget(&mut self) {
        self.bias = self.old_bias;
        self.weight1 = self.old_weight1;
        self.weight2 = self.old_weight2;
    }

    fn remember(&mut self) {
        self.old_bias = self.bias;
        self.old_weight1 = self.weight1;
        self.old_weight2 = self.weight2;
    }
}

fn sigmoid(input: f64) -> f64 {
    1_f64 / (1_f64 + std::f64::consts::E.powf(-input))
}
