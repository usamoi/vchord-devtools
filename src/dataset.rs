use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub d: usize,
    pub n: usize,
    pub m: usize,
    pub k: usize,
}
