mod chromosome;
mod utils;
mod graph;

use rand::{
    Rng,
    seq::SliceRandom,
};
use std::{
    collections::HashSet,
    borrow::BorrowMut,
};
use crate::chromosome::Chromosome;
use crate::utils::NonNanF32;
use crate::graph::Graph;


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

fn fill_graph_randomly(
    graph: &mut Graph,
    initial_probability: f32,
    rng: &mut impl Rng,
) -> f32 {
    let mut probability = initial_probability;
    loop {
        for i in 0..graph.vertices() {
            for j in i + 1..graph.vertices() {
                if rng.gen_range(0.0, 1.0) <= probability {
                    graph.set_edge(i, j, rng.gen_range(0, 10));
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
        let gene = rng.gen_range(0, ch.len() - 1);
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
        let gene1 = rng.gen_range(0, ch.len() - 1);
        let gene2 = rng.gen_range(0, ch.len() - 1);
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
        let gene1 = rng.gen_range(0, ch1.len() - 1);
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
        let gene2 = rng.gen_range(gene1 + 1, ch1.len() - 1);

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
    number_to_take: usize,
    fitness: impl Fn(&Specimen) -> f32,
    rng: &mut impl Rng,
) -> &'p Specimen {
    let mut best_specimen = None;

    for _ in 0..number_to_take {
        let current = rng.gen_range(0, population.len() - 1);
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
    f1: Option<NonNanF32>,
    f2: Option<NonNanF32>,
}

fn main() {
    let vertices = 64;
    let pop_size = 100;
    let mutation_probability = 0.315;
    let crossover_probability = 0.175;

    let mut rng = rand::thread_rng();


    let mut storage = vec![0; vertices * vertices];
    let mut graph = Graph::from_slice(vertices, storage.as_mut_slice());
    fill_graph_randomly(&mut graph, 0.00, &mut rng);

    for i in 0..vertices {
        for j in i + 1..vertices {
            let weight = graph.get_edge(i, j);
            if weight != 0 {
                println!("{} <--({})--> {}", i, weight, j);
            }
        }
    }

    println!("Calculating...");

    let mut population = initial_population(vertices, pop_size);
    let mut offspring = Vec::with_capacity(pop_size);

    for i in 0.. {
        for specimen in &mut population {
            let (f1, f2) = objective_functions(&graph, &specimen.chromosome);
            specimen.f1 = Some(NonNanF32(f1));
            specimen.f2 = Some(NonNanF32(f2));
        }

        let best1 = population.iter().min_by_key(|ch| ch.f1.unwrap()).unwrap();
        let best2 = population.iter().min_by_key(|ch| ch.f2.unwrap()).unwrap();

        println!("#{} {} {}", i, best1.f1.unwrap().0, best2.f2.unwrap().0);

        let (pop1, pop2) = population.split_at_mut(pop_size / 2);
        pop1.sort_by_key(|s| s.f1.unwrap());
        pop2.sort_by_key(|s| s.f2.unwrap());

        while offspring.len() < pop_size / 2 {
            let desc = tournament_succession(&pop1, 10, |s| -s.f1.unwrap().0, &mut rng);
            offspring.push(Specimen { chromosome: desc.chromosome.clone(), f1: None, f2: None });
        }

        while offspring.len() < pop_size {
            let desc = tournament_succession(&pop2, 10, |s| -s.f2.unwrap().0, &mut rng);
            offspring.push(Specimen { chromosome: desc.chromosome.clone(), f1: None, f2: None });
        }

        population.clear();
        std::mem::swap(&mut population, &mut offspring);
        population.shuffle(rand::thread_rng().borrow_mut());

        for p in &mut population {
            if rng.gen_range(0.0, 1.0) < mutation_probability {
                match rng.gen_range(0, 1) {
                    0 => single_mutation(&mut p.chromosome, &mut rng),
                    _ => replacement_mutation(&mut p.chromosome, &mut rng),
                }
            }
        }

        for p in population.chunks_mut(2) {
            if let [p1, p2] = p {
                if rng.gen_range(0.0, 1.0) < crossover_probability {
                    match rng.gen_range(0, 1) {
                        0 => onepoint_crossover(&mut p1.chromosome, &mut p2.chromosome, &mut rng),
                        _ => twopoint_crossover(&mut p1.chromosome, &mut p2.chromosome, &mut rng),
                    }
                }
            }
        }
    }
}

fn initial_population(vertices: usize, pop_size: usize) -> Vec<Specimen> {
    let mut population = Vec::with_capacity(pop_size);
    while population.len() < pop_size {
        let mut ch = Chromosome::with_length(vertices);
        for j in 0..vertices {
            ch.set(j, rand::random::<bool>().into());
        }

        if is_constraint_satisfied(&ch) {
            population.push(Specimen { chromosome: ch, f1: None, f2: None });
        }
    }
    population
}
