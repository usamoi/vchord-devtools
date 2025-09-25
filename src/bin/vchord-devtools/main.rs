mod import_hdf5;
mod load;

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(version, about, long_about = None)]
#[command(propagate_version = true)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    ImportHdf5(import_hdf5::Command),
    Load(load::Command),
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::ImportHdf5(command) => {
            import_hdf5::main(command)?;
        }
        Commands::Load(command) => {
            tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .build()?
                .block_on(load::main(command))?;
        }
    }

    Ok(())
}
