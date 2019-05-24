mod chromosome;
mod genetic_algorithm;
mod graph;

pub use crate::graph::{Graph, fill_graph_randomly, calculate_graph_density};
pub use crate::genetic_algorithm::{Config, IterationInfo, bipartition_ga, print_edges};
pub use crate::chromosome::Chromosome;