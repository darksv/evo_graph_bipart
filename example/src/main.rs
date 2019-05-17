use core::{bipartition_ga, Config, fill_graph_randomly, print_edges, Graph};
use rand::thread_rng;

fn main() {
    let vertices = 64;
    let mut rng = thread_rng();
    let mut storage = vec![0; vertices * vertices];
    let mut graph = Graph::from_slice(vertices, storage.as_mut_slice());
    fill_graph_randomly(&mut graph, 0.00, &mut rng);
    print_edges(vertices, &graph);
    bipartition_ga(&Config {
        population_size: 100,
        mutation_probability: 0.315,
        crossover_probability: 0.175,
        tournament_size: 10,
        max_iterations: Some(10000)
    }, &mut rng, &graph, |i, f1, f2| {
        println!("#{} {} {}", i, f1, f2);
    })
}
