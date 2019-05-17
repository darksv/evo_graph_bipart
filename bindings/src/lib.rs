use core::{fill_graph_randomly, Graph, bipartition_ga, Config};
use rand::thread_rng;
use std::borrow::BorrowMut;

#[no_mangle]
pub unsafe fn create_graph_instance(
    number_of_vertices: u32,
    storage: *mut u32,
    instance: *mut *mut Graph,
) -> u32 {
    if number_of_vertices < 1 {
        return 1;
    }

    if storage.is_null() {
        return 2;
    }

    if instance.is_null() {
        return 3;
    }

    let number_of_vertices = number_of_vertices as usize;
    let storage = std::slice::from_raw_parts_mut(storage, number_of_vertices * number_of_vertices);
    let mut graph = Box::new(Graph::from_slice(number_of_vertices, storage));
    *instance = &mut *graph as *mut Graph;
    std::mem::forget(graph);

    0
}

#[no_mangle]
pub unsafe fn destroy_graph_instance(
    instance: *mut Graph,
) -> u32 {
    if instance.is_null() {
        return 1;
    }

    let instance = Box::from_raw(instance);
    drop(instance);

    0
}

fn calculate_graph_density(graph: &Graph) -> f32 {
    let existing_edges = graph.edges() as f32;
    let vertices = graph.vertices() as f32;
    let all_edges = vertices * (vertices - 1.0) / 2.0;
    existing_edges / all_edges
}

#[no_mangle]
pub unsafe fn get_graph_density(
    instance: *const Graph,
    density: *mut f32,
) -> u32 {
    if instance.is_null() {
        return 1;
    }

    if density.is_null() {
        return 2;
    }

    *density = calculate_graph_density(&*instance);
    0
}

#[no_mangle]
pub unsafe fn randomize_graph(
    instance: *mut Graph,
    probability: *mut f32,
) -> u32 {
    if instance.is_null() {
        return 1;
    }

    *probability = fill_graph_randomly(&mut *instance, *probability, thread_rng().borrow_mut());
    0
}

#[no_mangle]
pub unsafe fn optimize_ga(
    instance: *const Graph,
    population_size: u32,
    mutation_probability: f32,
    crossover_probability: f32,
    iterations: u32,
    callback: fn(usize, f32, f32),
) -> u32 {
    if instance.is_null() {
        return 1;
    }

    if population_size == 0 {
        return 2;
    }

    if !(mutation_probability >= 0.0 && mutation_probability <= 1.0) {
        return 3;
    }

    if !(crossover_probability >= 0.0 && mutation_probability <= 1.0) {
        return 4;
    }

    if iterations == 0 {
        return 5;
    }

    bipartition_ga(&Config {
        population_size: population_size as usize,
        mutation_probability: mutation_probability as f64,
        crossover_probability: crossover_probability as f64,
        tournament_size: 0,
        max_iterations: None
    }, thread_rng().borrow_mut(), &*instance, callback);

    0
}