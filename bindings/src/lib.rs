use core::{fill_graph_randomly, calculate_graph_density, bipartition_ga, Graph, GeneticAlgorithmParameters, Chromosome, IterationInfo};
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

pub struct Handle {
    termination_tx: std::sync::mpsc::Sender<()>,
}

pub struct GaConfig {
    population_size: u32,
    mutation_probability: f32,
    crossover_probability: f32,
    tournament_size: u32,
    iterations: u32,
}

#[no_mangle]
pub unsafe fn run_genetic_optimization(
    instance: *const Graph<'static>,
    config: *const GaConfig,
    callback: fn(&IterationInfo),
    handle: *mut *const Handle,
) -> u32 {
    if instance.is_null() {
        return 1;
    }

    if config.is_null() {
        return 2;
    }

    let config = &*config;

    if config.population_size == 0 {
        return 3;
    }

    if !(config.mutation_probability >= 0.0 && config.mutation_probability <= 1.0) {
        return 4;
    }

    if !(config.crossover_probability >= 0.0 && config.mutation_probability <= 1.0) {
        return 5;
    }

    let max_iterations = if config.iterations == 0 {
        None
    } else {
        Some(config.iterations as usize)
    };

    let (tx, rx) = std::sync::mpsc::channel();
    let instance = &*instance;

    // Run a thread in the background that will wait for termination of the execution
    std::thread::spawn(move || {

        let params = GeneticAlgorithmParameters {
            population_size: config.population_size as usize,
            mutation_probability: config.mutation_probability as f64,
            crossover_probability: config.crossover_probability as f64,
            tournament_size: config.tournament_size as usize,
            max_iterations,
        };
        bipartition_ga(
            &params,
            thread_rng().borrow_mut(),
            instance,
            objective_functions,
            is_constraint_satisfied,
            |info| {
                callback(info);

                if rx.try_recv().is_ok() {
                    // Got a message - terminate execution
                    true
                } else {
                    // otherwise continue
                    false
                }
            },
        );
    });

    let new_handle = Box::new(Handle {
        termination_tx: tx,
    });

    *handle = &*new_handle as *const Handle;

    // We do not want to drop the sender...
    std::mem::forget(new_handle);

    0
}

#[no_mangle]
pub unsafe fn terminate_genetic_optimization(
    context: *mut Handle,
) -> u32 {
    if context.is_null() {
        return 1;
    }

    let ctx: Box<Handle> = Box::from_raw(context);
    ctx.termination_tx.send(()).unwrap();

    0
}

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