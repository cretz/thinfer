use core::marker::PhantomData;

pub trait ComputeDtype: 'static + Copy {
    const NAME: &'static str;
    const SIZE: usize;
}

#[derive(Clone, Copy)]
pub struct F32;
#[derive(Clone, Copy)]
pub struct F16;

impl ComputeDtype for F32 {
    const NAME: &'static str = "f32";
    const SIZE: usize = 4;
}
impl ComputeDtype for F16 {
    const NAME: &'static str = "f16";
    const SIZE: usize = 2;
}

#[derive(Clone, Debug)]
pub struct Shape(pub Vec<usize>);

impl Shape {
    pub fn rank(&self) -> usize {
        self.0.len()
    }
    pub fn elements(&self) -> usize {
        self.0.iter().product()
    }
}

#[derive(Clone, Debug)]
pub struct TensorDesc {
    pub shape: Shape,
    pub dtype: StorageEncoding,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageEncoding {
    F32,
    F16,
    Bf16,
    I8,
    I4,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GpuBufferId(pub u64);

pub struct GpuTensor<C: ComputeDtype> {
    pub buffer: GpuBufferId,
    pub shape: Shape,
    pub byte_offset: u64,
    _dtype: PhantomData<C>,
}

impl<C: ComputeDtype> GpuTensor<C> {
    pub fn new(buffer: GpuBufferId, shape: Shape, byte_offset: u64) -> Self {
        Self {
            buffer,
            shape,
            byte_offset,
            _dtype: PhantomData,
        }
    }
}

pub struct HostTensor {
    pub bytes: Vec<u8>,
    pub shape: Shape,
    pub encoding: StorageEncoding,
}
