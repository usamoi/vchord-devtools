use anyhow::Context;
use anyhow::bail;
use clap::Args;
use std::io::BufWriter;
use std::path::PathBuf;
use vchord_devtools::dataset::Manifest;
use vchord_devtools::vecs::Writer;

#[derive(Args)]
pub struct Command {
    #[arg(short, long)]
    pub input: PathBuf,
    #[arg(short, long)]
    pub output: PathBuf,
    #[arg(long)]
    pub block_size: Option<usize>,
    #[arg(short, long)]
    pub force: bool,
}

pub fn main(command: Command) -> anyhow::Result<()> {
    if !std::fs::exists(&command.input)? {
        bail!("input file does not exist");
    }
    if std::fs::exists(&command.output)? {
        if command.force {
            let _ = std::fs::remove_file(&command.output);
            std::fs::remove_dir_all(&command.output)?;
        } else {
            bail!("output directory already exists");
        }
    }
    let file = hdf5::File::open(command.input)?;
    let train = file.dataset("train")?;
    let test = file.dataset("test")?;
    let neighbors = file.dataset("neighbors")?;
    assert_eq!(train.shape().len(), 2);
    assert_eq!(test.shape().len(), 2);
    assert_eq!(neighbors.shape().len(), 2);
    let n = train.shape()[0];
    let d = train.shape()[1];
    let m = test.shape()[0];
    assert_eq!(d, test.shape()[1]);
    assert_eq!(m, neighbors.shape()[0]);
    let k = neighbors.shape()[1];
    assert!(train.dtype()?.is::<f32>());
    assert!(test.dtype()?.is::<f32>());
    assert!(
        neighbors.dtype()?.is::<i32>()
            || neighbors.dtype()?.is::<u32>()
            || neighbors.dtype()?.is::<i64>()
            || neighbors.dtype()?.is::<u64>()
    );
    std::fs::create_dir_all(&command.output)?;
    let mut options = std::fs::OpenOptions::new();
    options.read(true).write(true).append(true).create_new(true);
    {
        let mut writer = Writer::<_, f32>::new(BufWriter::new(
            options.open(command.output.join("train.fvecs"))?,
        ));
        let start = if let Some(block_size) = command.block_size {
            for i in 0..n / block_size {
                let r =
                    train.read_slice_2d::<f32, _>((i * block_size..(i + 1) * block_size, ..))?;
                for r in r.rows() {
                    writer.write(r.as_slice().context("failed to read hdf5")?)?;
                }
            }
            n / block_size * block_size
        } else {
            0
        };
        for i in start..n {
            let r = train.read_slice_1d::<f32, _>((i, ..))?;
            writer.write(r.as_slice().context("failed to read hdf5")?)?;
        }
    }
    {
        let mut writer = Writer::<_, f32>::new(BufWriter::new(
            options.open(command.output.join("test.fvecs"))?,
        ));
        for i in 0..m {
            let r = test.read_slice_1d::<f32, _>((i, ..))?;
            writer.write(r.as_slice().context("failed to read hdf5")?)?;
        }
    }
    {
        let mut writer = Writer::<_, i32>::new(BufWriter::new(
            options.open(command.output.join("groundtruth.ivecs"))?,
        ));
        for i in 0..m {
            let r = if neighbors.dtype()?.is::<i32>() {
                let r = neighbors.read_slice_1d::<i32, _>((i, ..))?;
                r.into_iter().map(|x| x as _).collect::<Vec<_>>()
            } else if neighbors.dtype()?.is::<u32>() {
                let r = neighbors.read_slice_1d::<u32, _>((i, ..))?;
                r.into_iter().map(|x| x as _).collect::<Vec<_>>()
            } else if neighbors.dtype()?.is::<i64>() {
                let r = neighbors.read_slice_1d::<i64, _>((i, ..))?;
                r.into_iter().map(|x| x as _).collect::<Vec<_>>()
            } else if neighbors.dtype()?.is::<u64>() {
                let r = neighbors.read_slice_1d::<u64, _>((i, ..))?;
                r.into_iter().map(|x| x as _).collect::<Vec<_>>()
            } else {
                unreachable!();
            };
            writer.write(r.as_slice())?;
        }
    }
    {
        let mut writer = BufWriter::new(options.open(command.output.join("manifest.json"))?);
        let manifest = Manifest { d, n, m, k };
        serde_json::to_writer(&mut writer, &manifest)?;
    }
    Ok(())
}
