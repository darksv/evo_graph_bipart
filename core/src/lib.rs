mod chromosome;
mod graph;

use rand::{Rng, seq::SliceRandom, thread_rng};
use std::{
    collections::HashSet,
    borrow::BorrowMut,
};
use crate::chromosome::Chromosome;
pub use crate::graph::Graph;
use rayon::iter::{IntoParallelRefMutIterator, ParallelIterator};
use rayon::slice::ParallelSliceMut;
use noisy_float::prelude::*;


fn objective_function1(
    graph: &Graph,
    ch: &Chromosome,
) -> f32 {
    graph.iter_connecting(ch).count() as f32
}

fn objective_function2(
    graph: &Graph,
    ch: &Chromosome,
) -> f32 {
    graph.iter_connecting(ch)
        .map(|(i, j)| graph.get_edge(i, j))
        .sum::<u32>() as f32
}

fn objective_functions(
    graph: &Graph,
    ch: &Chromosome,
) -> (f32, f32) {
    let (sum, count) = graph.iter_connecting(ch)
        .fold((0, 0), |(sum, count), (i, j)| {
            (sum + graph.get_edge(i, j), count + 1)
        });

    (sum as f32, count as f32)
}

fn is_constraint_satisfied(ch: &Chromosome) -> bool {
    let (v1, v2) = ch.count_genes_by_value();
    (v1 as i32 - v2 as i32).abs() <= 4
}

fn is_connected(graph: &Graph) -> bool {
    let mut visited = HashSet::new();
    let mut remaining = vec![0];

    while let Some(current) = remaining.pop() {
        if !visited.insert(current) {
            continue;
        }

        for j in 0..graph.vertices() {
            if graph.get_edge(current, j) != 0 {
                remaining.push(j);
            }
        }
    }

    visited.len() == graph.vertices()
}

pub fn fill_graph_randomly(
    graph: &mut Graph,
    initial_probability: f32,
    rng: &mut impl Rng,
) -> f32 {
    let mut probability = initial_probability;
    loop {
        for i in 0..graph.vertices() {
            for j in i + 1..graph.vertices() {
                if rng.gen_range(0.0, 1.0) <= probability {
                    graph.set_edge(i, j, rng.gen_range(0, 10 + 1));
                }
            }
        }

        if is_connected(graph) {
            return probability;
        } else {
            probability += 0.01;
            graph.clear();
        }
    }
}

fn single_mutation(
    ch: &mut Chromosome,
    rng: &mut impl Rng,
) {
    loop {
        let gene = rng.gen_range(0, ch.len());
        ch.toggle(gene);
        if is_constraint_satisfied(ch) {
            break;
        } else {
            ch.toggle(gene);
        }
    }
}

fn replacement_mutation(
    ch: &mut Chromosome,
    rng: &mut impl Rng,
) {
    loop {
        let gene1 = rng.gen_range(0, ch.len());
        let gene2 = rng.gen_range(0, ch.len());
        ch.swap(gene1, gene2);
        if is_constraint_satisfied(ch) {
            break;
        } else {
            ch.swap(gene1, gene2);
        }
    }
}

fn onepoint_crossover(
    ch1: &mut Chromosome,
    ch2: &mut Chromosome,
    rng: &mut impl Rng,
) {
    loop {
        let gene1 = rng.gen_range(0, ch1.len());
        for i in gene1..ch1.len() {
            std::mem::swap(ch1.get_mut(i), ch2.get_mut(i));
        }

        if is_constraint_satisfied(ch1) && is_constraint_satisfied(ch2) {
            break;
        } else {
            for i in gene1..ch1.len() {
                std::mem::swap(ch1.get_mut(i), ch2.get_mut(i));
            }
        }
    }
}

fn twopoint_crossover(
    ch1: &mut Chromosome,
    ch2: &mut Chromosome,
    rng: &mut impl Rng,
) {
    loop {
        let gene1 = rng.gen_range(0, ch1.len() - 1);
        let gene2 = rng.gen_range(gene1 + 1, ch1.len());

        for i in gene1..=gene2 {
            std::mem::swap(ch1.get_mut(i), ch2.get_mut(i));
        }

        if is_constraint_satisfied(ch1) && is_constraint_satisfied(ch2) {
            break;
        } else {
            // Restore...
            for i in gene1..=gene2 {
                std::mem::swap(ch1.get_mut(i), ch2.get_mut(i));
            }
        }
    }
}

fn tournament_succession<'p>(
    population: &'p [Specimen],
    selection_size: usize,
    fitness: impl Fn(&Specimen) -> f32,
    rng: &mut impl Rng,
) -> &'p Specimen {
    let mut best_specimen = None;

    for _ in 0..selection_size {
        let current = rng.gen_range(0, population.len());
        if let Some(best) = best_specimen {
            if fitness(&population[best]) >= fitness(&population[current]) {
                continue;
            }
        }
        best_specimen = Some(current);
    }

    &population[best_specimen.unwrap()]
}

struct Specimen {
    chromosome: Chromosome,
    f1: Option<N32>,
    f2: Option<N32>,
}

pub struct Config {
    pub population_size: usize,
    pub mutation_probability: f64,
    pub crossover_probability: f64,
    pub tournament_size: usize,
    pub max_iterations: Option<usize>,
}

pub fn bipartition_ga(
    config: &Config,
    rng: &mut impl Rng,
    graph: &Graph,
    callback: fn(usize, f32, f32),
) {
    let mut population = initial_population(graph.vertices(), config.population_size, rng);
    let mut offspring = Vec::with_capacity(config.population_size);

    let max_iterations = config.max_iterations.unwrap_or(usize::max_value());

    for i in 0..max_iterations {
        population.par_iter_mut().for_each(|specimen| {
            let (f1, f2) = objective_functions(&graph, &specimen.chromosome);
            specimen.f1 = Some(n32(f1));
            specimen.f2 = Some(n32(f2));
        });

        let best1 = population.iter().max_by_key(|ch| ch.f1.unwrap()).unwrap();
        let best2 = population.iter().max_by_key(|ch| ch.f2.unwrap()).unwrap();

        callback(i, best1.f1.unwrap().into(), best2.f2.unwrap().into());

        let (pop1, pop2) = population.split_at_mut(config.population_size / 2);
        pop1.par_sort_by_key(|s| s.f1.unwrap());
        pop2.par_sort_by_key(|s| s.f2.unwrap());

        while offspring.len() < config.population_size / 2 {
            let desc = tournament_succession(&pop1, config.tournament_size, |s| s.f1.unwrap().into(), rng);
            offspring.push(Specimen { chromosome: desc.chromosome.clone(), f1: None, f2: None });
        }

        while offspring.len() < config.population_size {
            let desc = tournament_succession(&pop2, config.tournament_size, |s| s.f2.unwrap().into(), rng);
            offspring.push(Specimen { chromosome: desc.chromosome.clone(), f1: None, f2: None });
        }

        population.clear();
        std::mem::swap(&mut population, &mut offspring);
        population.shuffle(rand::thread_rng().borrow_mut());

        population.par_iter_mut().for_each_init(|| rand::thread_rng(), |rng, p| {
            let should_mutate = rng.gen_range(0.0, 1.0) < config.mutation_probability;
            if !should_mutate {
                return;
            }

            let mutation = match rng.gen_range(0, 2) {
                0 => single_mutation,
                _ => replacement_mutation,
            };
            mutation(&mut p.chromosome, rng);
        });

        population.par_chunks_mut(2).for_each_init(|| rand::thread_rng(), |rng, pair| {
            let should_crossover = rng.gen_range(0.0, 1.0) < config.crossover_probability;
            if !should_crossover {
                return;
            }

            let (s1, s2) = match pair {
                [s1, s2] => (s1, s2),
                _ => return,
            };

            let crossover = match rng.gen_range(0, 2) {
                0 => onepoint_crossover,
                _ => twopoint_crossover,
            };
            crossover(&mut s1.chromosome, &mut s2.chromosome, rng);
        });
    }
}

pub fn print_edges(vertices: usize, graph: &Graph) {
    for i in 0..vertices {
        for j in i + 1..vertices {
            let weight = graph.get_edge(i, j);
            if weight != 0 {
                println!("{} <--({})--> {}", i, weight, j);
            }
        }
    }
}

fn initial_population(vertices: usize, pop_size: usize, rng: &mut impl Rng) -> Vec<Specimen> {
    let mut population = Vec::with_capacity(pop_size);
    while population.len() < pop_size {
        let mut ch = Chromosome::with_length(vertices);
        for j in 0..vertices {
            ch.set(j, rng.gen::<bool>().into());
        }

        if is_constraint_satisfied(&ch) {
            population.push(Specimen { chromosome: ch, f1: None, f2: None });
        }
    }
    population
}
