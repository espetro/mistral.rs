//! PointerHead port: logits = (k(h_opts) @ q(h_decide)) * scale / temperature.

use anyhow::Result;
use candle_core::{DType, Module, Tensor};
use candle_nn::Linear;

pub struct PointerHead {
    q: Linear,
    k: Linear,
    scale: f64,
    temperature: f64,
}

fn head_tensor(tensors: &safetensors::SafeTensors, name: &str) -> Result<Tensor> {
    let v = tensors.tensor(name)?;
    anyhow::ensure!(
        v.dtype() == safetensors::Dtype::F32,
        "head tensor {name} must be f32"
    );
    Ok(Tensor::from_raw_buffer(
        v.data(),
        DType::F32,
        v.shape(),
        &candle_core::Device::Cpu,
    )?)
}

fn head_linear(tensors: &safetensors::SafeTensors, prefix: &str) -> Result<Linear> {
    let w = head_tensor(tensors, &format!("{prefix}.weight"))?;
    let b = head_tensor(tensors, &format!("{prefix}.bias"))?;
    Ok(Linear::new(w, Some(b)))
}

impl PointerHead {
    pub fn load(path: &std::path::Path, head_dim: usize, temperature: f64) -> Result<Self> {
        let bytes = std::fs::read(path)?;
        let tensors = safetensors::SafeTensors::deserialize(&bytes)?;
        Ok(Self {
            q: head_linear(&tensors, "q")?,
            k: head_linear(&tensors, "k")?,
            scale: 1.0 / (head_dim as f64).sqrt(),
            temperature,
        })
    }

    /// [d] decide hidden, [K, d] option hidden -> logits [K].
    pub fn logits(&self, h_decide: &Tensor, h_opts: &Tensor) -> Result<Vec<f32>> {
        let q = self
            .q
            .forward(&h_decide.to_dtype(DType::F32)?.unsqueeze(0)?)?;
        let ks = self.k.forward(&h_opts.to_dtype(DType::F32)?)?;
        let z = ks
            .matmul(&q.t()?)?
            .squeeze(1)?
            .affine(self.scale / self.temperature, 0.0)?;
        Ok(z.to_vec1::<f32>()?)
    }
}

/// Softmax over one logits vector, matching F.softmax.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - m).exp()).collect();
    let s: f32 = exps.iter().sum();
    exps.iter().map(|x| x / s).collect()
}
