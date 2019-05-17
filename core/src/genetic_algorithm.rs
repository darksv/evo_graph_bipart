use rand::{Rng, seq::SliceRandom};
use noisy_float::prelude::*;
use rayon::prelude::*;
use crate::{Graph, chromosome::Chromosome};

pub trait Constraint: Copy {
    fn is_satisfied(&self, chromosome: &Chromosome) -> bool;
}

impl<F> Constraint for F
    where
        F: Fn(&Chromosome) -> bool + Copy
{
    #[inline]
    fn is_satisfied(&self, chromosome: &Chromosome) -> bool {
        self(chromosome)
    }
}

pub trait ObjectiveFunction: Send + Sync + Copy {
    fn calculate(&self, graph: &Graph, chromosome: &Chromosome) -> (f32, f32);
}

impl<F> ObjectiveFunction for F
    where
        F: Fn(&Graph, &Chromosome) -> (f32, f32) + Copy + Send + Sync
{
    #[inline]
    fn calculate(&self, graph: &Graph, chromosome: &Chromosome) -> (f32, f32) {
        self(graph, chromosome)
    }
}

fn single_mutation(
    ch: &mut Chromosome,
    constraint: impl Constraint,
    rng: &mut impl Rng,
) {
    loop {
        let gene = rng.gen_range(0, ch.len());
        ch.toggle(gene);
        if constraint.is_satisfied(ch) {
            break;
        } else {
            ch.toggle(gene);
        }
    }
}

fn replacement_mutation(
    ch: &mut Chromosome,
    constraint: impl Constraint,
    rng: &mut impl Rng,
) {
    loop {
        let gene1 = rng.gen_range(0, ch.len());
        let gene2 = rng.gen_range(0, ch.len());
        ch.swap(gene1, gene2);
        if constraint.is_satisfied(ch) {
            break;
        } else {
            ch.swap(gene1, gene2);
        }
    }
}

fn onepoint_crossover(
    ch1: &mut Chromosome,
    ch2: &mut Chromosome,
    constraint: impl Constraint,
    rng: &mut impl Rng,
) {
    loop {
        let gene1 = rng.gen_range(0, ch1.len());
        for i in gene1..ch1.len() {
            std::mem::swap(ch1.get_mut(i), ch2.get_mut(i));
        }

        if constraint.is_satisfied(ch1) && constraint.is_satisfied(ch2) {
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
    constraint: impl Constraint,
    rng: &mut impl Rng,
) {
    loop {
        let gene1 = rng.gen_range(0, ch1.len() - 1);
        let gene2 = rng.gen_range(gene1 + 1, ch1.len());

        for i in gene1..=gene2 {
            std::mem::swap(ch1.get_mut(i), ch2.get_mut(i));
        }

        if constraint.is_satisfied(ch1) && constraint.is_satisfied(ch2) {
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
    fitness: fn(&Specimen) -> f32,
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
    objectives: impl ObjectiveFunction,
    constraint: impl Constraint,
    callback: fn(usize, f32, f32),
) {
    let mut population = initial_population(graph.vertices(), config.population_size, constraint, rng);
    let mut offspring = Vec::with_capacity(config.population_size);

    let max_iterations = config.max_iterations.unwrap_or(usize::max_value());

    for i in 0..max_iterations {
        population.par_iter_mut().for_each(|specimen: &mut Specimen| {
            let (f1, f2) = objectives.calculate(&graph, &specimen.chromosome);
            specimen.f1 = Some(n32(f1));
            specimen.f2 = Some(n32(f2));
        });

        let best1 = population.iter().max_by_key(|ch| ch.f1.unwrap()).unwrap();
        let best2 = population.iter().max_by_key(|ch| ch.f2.unwrap()).unwrap();

        callback(i, best1.f1.unwrap().into(), best2.f2.unwrap().into());

        let (pop1, pop2) = population.split_at_mut(config.population_size / 2);
        pop1.sort_by_key(|s| s.f1.unwrap());
        pop2.sort_by_key(|s| s.f2.unwrap());

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
        population.shuffle(rng);

        for specimen in population.iter_mut() {
            let should_mutate = rng.gen_range(0.0, 1.0) < config.mutation_probability;
            if !should_mutate {
                continue;
            }

            let mutation = match rng.gen_range(0, 2) {
                0 => single_mutation,
                _ => replacement_mutation,
            };
            mutation(&mut specimen.chromosome, constraint, rng);
        }

        for pair in population.chunks_exact_mut(2) {
            let should_crossover = rng.gen_range(0.0, 1.0) < config.crossover_probability;
            if !should_crossover {
                continue;
            }

            let (s1, s2) = match pair {
                [s1, s2] => (s1, s2),
                _ => continue,
            };

            let crossover = match rng.gen_range(0, 2) {
                0 => onepoint_crossover,
                _ => twopoint_crossover,
            };
            crossover(&mut s1.chromosome, &mut s2.chromosome, constraint, rng);
        }
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

fn initial_population(
    vertices: usize,
    population_size: usize,
    constraint: impl Constraint,
    rng: &mut impl Rng,
) -> Vec<Specimen> {
    let mut population = Vec::with_capacity(population_size);
    while population.len() < population_size {
        let mut ch = Chromosome::with_length(vertices);
        for j in 0..vertices {
            ch.set(j, rng.gen::<bool>().into());
        }

        if constraint.is_satisfied(&ch) {
            population.push(Specimen { chromosome: ch, f1: None, f2: None });
        }
    }
    population
}