use anyhow::Context;
use clap::Args;
use futures::SinkExt;
use std::pin::pin;
use std::{collections::VecDeque, path::PathBuf};
use vchord_devtools::{dataset::Manifest, vecs::AsyncReader};

#[derive(Args)]
pub struct Command {
    #[arg(short, long)]
    pub input: PathBuf,
    #[arg(short, long)]
    pub force: bool,
}

const SIGN: [u8; 11] = [
    b'P', b'G', b'C', b'O', b'P', b'Y', b'\n', 255, b'\r', b'\n', 0,
];

pub async fn main(command: Command) -> anyhow::Result<()> {
    let name = command
        .input
        .canonicalize()?
        .file_name()
        .context("input is a root dir")?
        .to_str()
        .context("input has a bad name")?
        .to_string();
    let (client, conn) = tokio_postgres::connect(
        &format!("postgres://{}@localhost", whoami::username()),
        tokio_postgres::NoTls,
    )
    .await?;
    tokio::spawn(conn);
    let manifest: Manifest = serde_json::from_str(&std::fs::read_to_string(
        command.input.join("manifest.json"),
    )?)?;
    {
        let mut f_train = AsyncReader::<_, f32>::new(tokio::io::BufReader::new(
            tokio::fs::File::open(command.input.join("train.fvecs")).await?,
        ));
        if command.force {
            client
                .execute(&format!("DROP TABLE IF EXISTS {name}_train"), &[])
                .await?;
        }
        client
            .execute(
                &format!(
                    "CREATE TABLE {name}_train(index integer, embedding vector({}))",
                    manifest.d
                ),
                &[],
            )
            .await?;
        let mut buf = pin!(
            client
                .copy_in::<_, VecDeque<u8>>(&format!(
                    "COPY {name}_train (index, embedding) FROM STDIN WITH (FORMAT BINARY)"
                ))
                .await?
        );
        buf.feed(SIGN.into()).await?;
        buf.feed(0i32.to_be_bytes().into()).await?;
        buf.feed(0i32.to_be_bytes().into()).await?;
        for i in 0..manifest.n {
            let index = i as i32;
            let embedding = f_train
                .read()
                .await?
                .context("number of train vectors is not expected")?;
            if embedding.len() != manifest.d {
                anyhow::bail!("dimension of train vectors is not expected");
            }
            let mut buffer = Vec::<u8>::new();
            buffer.extend(2i16.to_be_bytes());

            buffer.extend([0u8; 4]);
            let field_start = buffer.len();
            // ******
            buffer.extend(index.to_be_bytes());
            // ******
            let field_len = (buffer.len() - field_start) as i32;
            buffer[field_start - 4 + 0] = field_len.to_be_bytes()[0];
            buffer[field_start - 4 + 1] = field_len.to_be_bytes()[1];
            buffer[field_start - 4 + 2] = field_len.to_be_bytes()[2];
            buffer[field_start - 4 + 3] = field_len.to_be_bytes()[3];

            buffer.extend([0u8; 4]);
            let field_start = buffer.len();
            // ******
            buffer.extend((embedding.len() as i16).to_be_bytes());
            buffer.extend(0_i16.to_be_bytes());
            buffer.extend(embedding.iter().flat_map(|x| x.to_be_bytes()));
            // ******
            let field_len = (buffer.len() - field_start) as i32;
            buffer[field_start - 4 + 0] = field_len.to_be_bytes()[0];
            buffer[field_start - 4 + 1] = field_len.to_be_bytes()[1];
            buffer[field_start - 4 + 2] = field_len.to_be_bytes()[2];
            buffer[field_start - 4 + 3] = field_len.to_be_bytes()[3];

            buf.feed(buffer.into()).await?;
        }
        if f_train.read().await?.is_some() {
            anyhow::bail!("number of train vectors is not expected");
        }
        buf.flush().await?;
        buf.finish().await?;
    }
    {
        let mut f_test = AsyncReader::<_, f32>::new(tokio::io::BufReader::new(
            tokio::fs::File::open(command.input.join("test.fvecs")).await?,
        ));
        let mut f_groundtruth = AsyncReader::<_, i32>::new(tokio::io::BufReader::new(
            tokio::fs::File::open(command.input.join("groundtruth.ivecs")).await?,
        ));
        if command.force {
            client
                .execute(&format!("DROP TABLE IF EXISTS {name}_test"), &[])
                .await?;
        }
        client
        .execute(
            &format!(
                "CREATE TABLE {name}_test(index integer, embedding vector({}), answer integer[])",
                manifest.d
            ),
            &[],
        )
        .await?;
        let mut buf =
            pin!(client
            .copy_in::<_, VecDeque<u8>>(&format!(
                "COPY {name}_test (index, embedding, answer) FROM STDIN WITH (FORMAT BINARY)"
            ))
            .await?);
        buf.feed(SIGN.into()).await?;
        buf.feed(0i32.to_be_bytes().into()).await?;
        buf.feed(0i32.to_be_bytes().into()).await?;
        for i in 0..manifest.m {
            let index = i as i32;
            let embedding = f_test
                .read()
                .await?
                .context("number of test vectors is not expected")?;
            let answer = f_groundtruth
                .read()
                .await?
                .context("number of test neighbours is not expected")?;
            if embedding.len() != manifest.d {
                anyhow::bail!("dimension of test vectors is not expected");
            }
            if answer.len() != manifest.k {
                anyhow::bail!("dimension of test neighbours is not expected");
            }
            let mut buffer = Vec::<u8>::new();
            buffer.extend(3i16.to_be_bytes());

            buffer.extend([0u8; 4]);
            let field_start = buffer.len();
            // ******
            buffer.extend(index.to_be_bytes());
            // ******
            let field_len = (buffer.len() - field_start) as i32;
            buffer[field_start - 4 + 0] = field_len.to_be_bytes()[0];
            buffer[field_start - 4 + 1] = field_len.to_be_bytes()[1];
            buffer[field_start - 4 + 2] = field_len.to_be_bytes()[2];
            buffer[field_start - 4 + 3] = field_len.to_be_bytes()[3];

            buffer.extend([0u8; 4]);
            let field_start = buffer.len();
            // ******
            buffer.extend((embedding.len() as i16).to_be_bytes());
            buffer.extend(0_i16.to_be_bytes());
            buffer.extend(embedding.iter().flat_map(|x| x.to_be_bytes()));
            // ******
            let field_len = (buffer.len() - field_start) as i32;
            buffer[field_start - 4 + 0] = field_len.to_be_bytes()[0];
            buffer[field_start - 4 + 1] = field_len.to_be_bytes()[1];
            buffer[field_start - 4 + 2] = field_len.to_be_bytes()[2];
            buffer[field_start - 4 + 3] = field_len.to_be_bytes()[3];

            // start of answer
            buffer.extend([0u8; 4]);
            let field_start = buffer.len();
            // ******
            buffer.extend(1_i32.to_be_bytes()); // ndim
            buffer.extend(0_i32.to_be_bytes()); // has_null
            buffer.extend(23_i32.to_be_bytes()); // element type
            buffer.extend((answer.len() as i32).to_be_bytes()); // d[0]
            buffer.extend(1_i32.to_be_bytes()); // l[0]
            for x in answer.iter().copied() {
                buffer.extend(4_i32.to_be_bytes());
                buffer.extend(x.to_be_bytes());
            }
            // ******
            let field_len = (buffer.len() - field_start) as i32;
            buffer[field_start - 4 + 0] = field_len.to_be_bytes()[0];
            buffer[field_start - 4 + 1] = field_len.to_be_bytes()[1];
            buffer[field_start - 4 + 2] = field_len.to_be_bytes()[2];
            buffer[field_start - 4 + 3] = field_len.to_be_bytes()[3];

            buf.feed(buffer.into()).await?;
        }
        if f_test.read().await?.is_some() {
            anyhow::bail!("number of test vectors is not expected");
        }
        if f_groundtruth.read().await?.is_some() {
            anyhow::bail!("number of test neighbours is not expected");
        }
        buf.flush().await?;
        buf.finish().await?;
    }
    Ok(())
}
