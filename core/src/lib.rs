mod chromosome;
mod genetic_algorithm;
mod graph;

pub use crate::graph::Graph;
pub use crate::genetic_algorithm::{Config, IterationInfo, bipartition_ga, print_edges};
pub use crate::graph::fill_graph_randomly;
pub use crate::chromosome::Chromosome;