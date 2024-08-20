use std::time::{Duration, Instant};
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Tensor};
use tch::nn::{Optimizer, Sequential};
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::whitespace::Whitespace;
use tokenizers::decoders::bpe::BPEDecoder;
use tokenizers::processors::byte_level::ByteLevel;
use tokenizers::{Tokenizer, TokenizerBuilder};
use std::fs;
use tch::nn::ModuleT;
use std::any::Any;

#[derive(Debug)]
pub struct QuinaryLinear {
    pub ws: Tensor,
    pub bs: Option<Tensor>,
}

impl QuinaryLinear {
    fn new(vs: &nn::Path, in_dim: i64, out_dim: i64) -> Self {
        let ws = vs.randn("ws", &[out_dim, in_dim], 0.0, 0.02);
        let bs = vs.randn("bs", &[out_dim], 0.0, 0.02);
        QuinaryLinear {
            ws,
            bs: Some(bs),
        }
    }

    fn forward(&self, xs: &Tensor) -> Tensor {
        let ws_quantized = (self.ws * 15.0).round() / 15.0;
        xs.matmul(&ws_quantized) + self.bs.as_ref().unwrap()
    }
}

impl nn::Module for QuinaryLinear {
    fn forward(&self, xs: &Tensor) -> Tensor {
        self.forward(xs)
    }
}

impl AsAny for QuinaryLinear {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub trait AsAny {
    fn as_any(&self) -> &dyn Any;
}

struct Func<F>(F);

impl<F> nn::Module for Func<F>
where
    F: 'static + Fn(&Tensor) -> Tensor + Send,
{
    fn forward(&self, xs: &Tensor) -> Tensor {
        (self.0)(xs)
    }
}

impl<F> AsAny for Func<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct FuncT<F>(F);

impl<F> nn::ModuleT for FuncT<F>
where
    F: 'static + Fn(&Tensor, bool) -> Tensor + Send,
{
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        (self.0)(xs, train)
    }
}

impl<F> AsAny for FuncT<F> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

// Sequential and SequentialT implementations

/// A sequential layer combining multiple other layers.
#[derive(Debug)]
pub struct MySequential {
    layers: Vec<Box<dyn Module>>,
}

/// Creates a new empty sequential layer.
pub fn seq() -> MySequential {
    MySequential { layers: vec![] }
}

impl MySequential {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Provides access to the layers.
    pub fn layers(&self) -> &Vec<Box<dyn Module>> {
        &self.layers
    }
}

impl Module for MySequential {
    fn forward(&self, xs: &Tensor) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            let xs = self.layers[0].forward(xs);
            self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward(&xs))
        }
    }
}

impl MySequential {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: Module + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Tensor + Send,
    {
        self.add(Func(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all(&self, xs: &Tensor, n: Option<usize>) -> Vec<Tensor> {
        if self.layers.is_empty() {
            vec![xs.shallow_clone()]
        } else {
            let n = n.unwrap_or(self.layers.len());
            let xs = self.layers[0].forward(xs);
            let mut vec = vec![];
            let out = self.layers.iter().take(n).skip(1).fold(xs, |xs, layer| {
                let out = layer.forward(&xs);
                vec.push(xs);
                out
            });
            vec.push(out);
            vec
        }
    }
}

/// A sequential layer combining new layers with support for a training mode.
#[derive(Debug)]
pub struct MySequentialT {
    layers: Vec<Box<dyn ModuleT>>,
}

/// Creates a new empty sequential layer.
pub fn seq_t() -> MySequentialT {
    MySequentialT { layers: vec![] }
}

impl MySequentialT {
    /// The number of sub-layers embedded in this layer.
    pub fn len(&self) -> i64 {
        self.layers.len() as i64
    }

    /// Returns true if this layer does not have any sub-layer.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Provides access to the layers.
    pub fn layers(&self) -> &Vec<Box<dyn ModuleT>> {
        &self.layers
    }
}

impl ModuleT for MySequentialT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        if self.layers.is_empty() {
            xs.shallow_clone()
        } else {
            let xs = self.layers[0].forward_t(xs, train);
            self.layers.iter().skip(1).fold(xs, |xs, layer| layer.forward_t(&xs, train))
        }
    }
}

impl MySequentialT {
    /// Appends a layer after all the current layers.
    #[allow(clippy::should_implement_trait)]
    pub fn add<M: ModuleT + 'static>(mut self, layer: M) -> Self {
        self.layers.push(Box::new(layer));
        self
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor) -> Tensor + Send,
    {
        self.add(Func(f))
    }

    /// Appends a closure after all the current layers.
    pub fn add_fn_t<F>(self, f: F) -> Self
    where
        F: 'static + Fn(&Tensor, bool) -> Tensor + Send,
    {
        self.add(FuncT(f))
    }

    /// Applies the forward pass and returns the output for each layer.
    pub fn forward_all_t(&self, xs: &Tensor, train: bool, n: Option<usize>) -> Vec<Tensor> {
        if self.layers.is_empty() {
            vec![xs.shallow_clone()]
        } else {
            let n = n.unwrap_or(self.layers.len());
            let xs = self.layers[0].forward_t(xs, train);
            let mut vec = vec![];
            let out = self.layers.iter().take(n).skip(1).fold(xs, |xs, layer| {
                let out = layer.forward_t(&xs, train);
                vec.push(xs);
                out
            });
            vec.push(out);
            vec
        }
    }
}

fn replace_gpt2_linear_layers(seq: &MySequential, vs: &nn::Path) -> MySequential {
    let mut new_seq = seq();
    for layer in seq.layers().iter() {
        if let Some(linear) = layer.as_any().downcast_ref::<nn::Linear>() {
            let in_features = linear.ws.size()[1];
            let out_features = linear.ws.size()[0];
            let quinary_layer = QuinaryLinear::new(vs, in_features, out_features);
            new_seq = new_seq.add(quinary_layer);
        } else {
            new_seq = new_seq.add(layer.clone());
        }
    }
    new_seq
}


fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    // Initialize the tokenizer
    let tokenizer = TokenizerBuilder::new()
        .with_model(BPE::default())
        .with_pre_tokenizer(Some(Whitespace::default()))
        .with_decoder(Some(BPEDecoder::default()))
        .with_post_processor(Some(ByteLevel::default()))
        .build()?;

    // Create a GPT-2 configuration
    let vs = nn::VarStore::new(Device::cuda_if_available());

    // Initialize the model
    let mut model = seq();
    model = replace_gpt2_linear_layers(&model, &vs.root());

    // Set up the optimizer
    let mut opt = nn::Adam::default().build(&vs, 1e-4)?;

    // Load weights if checkpoint exists
    let checkpoint_path = "model_checkpoint.ot";
    if fs::metadata(checkpoint_path).is_ok() {
        vs.load(checkpoint_path)?;
        println!("Loaded checkpoint from {}", checkpoint_path);
    }

    // Training loop with time measurement per batch
    let mut batch_times: Vec<Duration> = Vec::new();
    for epoch in 1..4 {
        for batch_idx in 0..100 {
            let start_time = Instant::now();

            // Placeholder tensor for xs and ys
            let xs = Tensor::randn(&[64, 512], (tch::Kind::Float, vs.device()));
            let ys = Tensor::randn(&[64, 512], (tch::Kind::Float, vs.device()));

            let loss = model.forward(&xs).mse_loss(&ys, tch::Reduction::Mean);

            opt.backward_step(&loss);

            let batch_time = start_time.elapsed();
            batch_times.push(batch_time);

            println!("Epoch: {}, Batch: {}, Loss: {:?}", epoch, batch_idx, f64::from(loss));

            // Save checkpoint every 100 batches
            if batch_idx % 100 == 0 {
                vs.save(checkpoint_path)?;
                println!("Saved checkpoint at batch {}", batch_idx);
            }
        }
    }

    // Calculate average batch time
    let total_batch_time: Duration = batch_times.iter().sum();
    let average_batch_time = total_batch_time / (batch_times.len() as u32);

    // Calculate total training time
    let total_batches = 100 * 3;
    let total_training_time = average_batch_time * (total_batches as u32);

    println!("Average batch time: {:.2?}", average_batch_time);
    println!("Estimated total training time: {:.2?}", total_training_time);

    // Conversational model usage
    loop {
        let mut input_text = String::new();
        std::io::stdin().read_line(&mut input_text)?;
        let inputs = tokenizer.encode(input_text.trim(), true)?;
        let inputs_tensor = Tensor::from_slice(&inputs.get_ids()).view([-1, 1]).to(vs.device());
        let outputs = model.forward(&inputs_tensor).softmax(-1, tch::Kind::Float);
        let response = tokenizer.decode(outputs.argmax(-1, false).tolist(), true)?;
        println!("Bot: {}", response);
    }

    Ok(())
}
