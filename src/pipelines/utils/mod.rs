use crate::core::ModelOptions;
use candle_core::{CudaDevice, Device};

// Re-export builder utilities
pub mod builder;
pub use builder::{BasePipelineBuilder, StandardPipelineBuilder};

/// Loads a device to be used for the model.
/// If `index` is `Some(i)` it will attempt to load the specified CUDA device.
/// When `None` it defaults to CUDA device 0 if available and otherwise falls back
/// to CPU.
pub fn load_device_with(index: Option<usize>) -> anyhow::Result<Device> {
    if let Some(i) = index {
        let cuda = CudaDevice::new_with_stream(i)?;
        Ok(Device::Cuda(cuda))
    } else {
        match CudaDevice::new_with_stream(0) {
            Ok(cuda_device) => Ok(Device::Cuda(cuda_device)),
            Err(_) => Ok(Device::Cpu),
        }
    }
}

/// Convenience wrapper that preserves the previous behaviour of selecting CUDA 0
/// if available and otherwise falling back to CPU.
pub fn load_device() -> anyhow::Result<Device> {
    load_device_with(None)
}

/// Request for a specific device, used by pipeline builders.
#[derive(Clone, Default)]
pub enum DeviceRequest {
    /// Use CUDA if available, otherwise CPU (default behavior).
    #[default]
    Default,
    /// Force CPU even if CUDA is available.
    Cpu,
    /// Select a specific CUDA device by index.
    Cuda(usize),
    /// Provide an already constructed device.
    Explicit(Device),
}

impl DeviceRequest {
    /// Resolve the request into an actual [`Device`].
    pub fn resolve(self) -> anyhow::Result<Device> {
        match self {
            DeviceRequest::Default => load_device_with(None),
            DeviceRequest::Cpu => Ok(Device::Cpu),
            DeviceRequest::Cuda(i) => Ok(Device::Cuda(CudaDevice::new_with_stream(i)?)),
            DeviceRequest::Explicit(d) => Ok(d),
        }
    }
}

/// Trait providing convenience methods for pipeline builders to select a device.
pub trait DeviceSelectable: Sized {
    /// Returns a mutable reference to the builder's internal [`DeviceRequest`].
    fn device_request_mut(&mut self) -> &mut DeviceRequest;

    /// Force the pipeline to run on CPU.
    fn cpu(mut self) -> Self {
        *self.device_request_mut() = DeviceRequest::Cpu;
        self
    }

    /// Select a specific CUDA device by index.
    fn cuda_device(mut self, index: usize) -> Self {
        *self.device_request_mut() = DeviceRequest::Cuda(index);
        self
    }

    /// Provide an explicit [`Device`].
    fn device(mut self, device: Device) -> Self {
        *self.device_request_mut() = DeviceRequest::Explicit(device);
        self
    }
}

/// Utility to generate a cache key combining model options and device location.
pub fn build_cache_key<O: ModelOptions>(options: &O, device: &Device) -> String {
    format!("{}-{:?}", options.cache_key(), device.location())
}
