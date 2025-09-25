use byteorder::WriteBytesExt;
use byteorder::{LE, ReadBytesExt};
use std::io::{BufRead, Write};
use std::marker::PhantomData;
use thiserror::Error;
use tokio::io::AsyncBufRead;
use tokio::io::AsyncBufReadExt;
use tokio::io::AsyncReadExt;
use tokio::io::AsyncWrite;
use tokio::io::AsyncWriteExt;
use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes};

mod sealed {
    pub trait Sealed {}

    impl Sealed for u8 {}

    impl Sealed for i32 {}

    impl Sealed for f32 {}
}

pub trait Scalar: sealed::Sealed + Copy + FromZeros + FromBytes + IntoBytes + Immutable {}

impl Scalar for u8 {}

impl Scalar for i32 {}

impl Scalar for f32 {}

#[derive(Debug, Error)]
pub enum VecsError {
    #[error("io error")]
    Io(#[from] std::io::Error),
    #[error("data corruption")]
    Corruption,
    #[error("too big")]
    TooBig,
}

pub struct Reader<R, T> {
    read: R,
    _phantom: PhantomData<T>,
}

impl<R: BufRead, T: Scalar> Reader<R, T> {
    pub fn new(read: R) -> Self {
        Self {
            read,
            _phantom: PhantomData,
        }
    }
    pub fn read(&mut self) -> Result<Option<Vec<T>>, VecsError> {
        if !self.read.fill_buf().map(|b| !b.is_empty())? {
            return Ok(None);
        }
        let d = self.read.read_i32::<LE>()?;
        if d < 0 {
            return Err(VecsError::Corruption);
        }
        let mut buffer = vec![T::new_zeroed(); d as usize];
        self.read.read_exact(buffer.as_mut_bytes())?;
        Ok(Some(buffer))
    }
}

pub struct AsyncReader<R, T> {
    read: R,
    _phantom: PhantomData<T>,
}

impl<R: AsyncBufRead + Unpin, T: Scalar> AsyncReader<R, T> {
    pub fn new(read: R) -> Self {
        Self {
            read,
            _phantom: PhantomData,
        }
    }
    pub async fn read(&mut self) -> Result<Option<Vec<T>>, VecsError> {
        if !self.read.fill_buf().await.map(|b| !b.is_empty())? {
            return Ok(None);
        }
        let d = self.read.read_i32_le().await?;
        if d < 0 {
            return Err(VecsError::Corruption);
        }
        let mut buffer = vec![T::new_zeroed(); d as usize];
        self.read.read_exact(buffer.as_mut_bytes()).await?;
        Ok(Some(buffer))
    }
}

pub struct Writer<R, T> {
    write: R,
    _phantom: PhantomData<T>,
}

impl<W: Write, T: Scalar> Writer<W, T> {
    pub fn new(write: W) -> Self {
        Self {
            write,
            _phantom: PhantomData,
        }
    }
    pub fn write(&mut self, data: &[T]) -> Result<(), VecsError> {
        let d = data.len();
        if d >= i32::MAX as usize {
            return Err(VecsError::TooBig);
        }
        self.write.write_i32::<LE>(d as _)?;
        self.write.write_all(data.as_bytes())?;
        Ok(())
    }
}

pub struct AsyncWriter<R, T> {
    write: R,
    _phantom: PhantomData<T>,
}

impl<W: AsyncWrite + Unpin, T: Scalar> AsyncWriter<W, T> {
    pub fn new(write: W) -> Self {
        Self {
            write,
            _phantom: PhantomData,
        }
    }
    pub async fn write(&mut self, data: &[T]) -> Result<(), VecsError> {
        let d = data.len();
        if d >= i32::MAX as usize {
            return Err(VecsError::TooBig);
        }
        self.write.write_i32_le(d as _).await?;
        self.write.write_all(data.as_bytes()).await?;
        Ok(())
    }
}
