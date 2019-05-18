use core::{bipartition_ga, Config, fill_graph_randomly, print_edges, Graph, Chromosome};
use rand::thread_rng;


fn objective_functions(
    graph: &Graph,
    ch: &Chromosome,
) -> (f32, f32) {
    let (sum, count) = graph.iter_connecting(ch)
        .fold((0, 0), |(sum, count), edge| {
            (sum + edge.weight, count + 1)
        });
    (sum as f32, count as f32)
}

fn is_constraint_satisfied(ch: &Chromosome) -> bool {
    let (v1, v2) = ch.count_genes_by_value();
    (v1 as i32 - v2 as i32).abs() <= 4
}


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
    }, &mut rng, &graph, objective_functions,is_constraint_satisfied,|i, f1, f2| {
        println!("#{} {} {}", i, f1, f2);
    });
}
