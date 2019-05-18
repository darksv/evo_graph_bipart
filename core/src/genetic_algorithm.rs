use rand::{Rng, seq::SliceRandom};
use noisy_float::prelude::*;
use rayon::prelude::*;
use crate::{Graph, chromosome::{Chromosome, Gene}};

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

fn tournament_succession<'pool>(
    population: &'pool [Specimen],
    selection_size: usize,
    fitness: fn(&Specimen) -> f32,
    rng: &mut impl Rng,
) -> &'pool Specimen<'pool> {
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

struct Specimen<'a> {
    chromosome: Chromosome<'a>,
    f1: Option<N32>,
    f2: Option<N32>,
}

impl<'a> Specimen<'a> {
    fn from_chromosome(chromosome: Chromosome) -> Specimen {
        Specimen {
            chromosome,
            f1: Option::None,
            f2: Option::None
        }
    }
}

pub struct Config {
    pub population_size: usize,
    pub mutation_probability: f64,
    pub crossover_probability: f64,
    pub tournament_size: usize,
    pub max_iterations: Option<usize>,
}

struct PopulationPool {
    data: Vec<Gene>,
    number_of_specimens: usize,
    chromosome_length: usize,
}

impl PopulationPool {
    fn new(number_of_specimens: usize, size_of_chromosome: usize) -> Self {
        Self {
            data: vec![Gene::Zero; number_of_specimens * size_of_chromosome],
            number_of_specimens,
            chromosome_length: size_of_chromosome,
        }
    }

    fn len(&self) -> usize {
        self.number_of_specimens
    }

    fn nth_mut(&mut self, n: usize) -> &mut [Gene] {
        self.data.chunks_exact_mut(self.chromosome_length).nth(n).unwrap()
    }

    fn chromosomes(&self) -> impl Iterator<Item=&[Gene]> {
        self.data.chunks_exact(self.chromosome_length)
    }

    fn chromosomes_mut(&mut self) -> impl Iterator<Item=&mut [Gene]> {
        self.data.chunks_exact_mut(self.chromosome_length)
    }
}

pub fn bipartition_ga(
    config: &Config,
    rng: &mut impl Rng,
    graph: &Graph,
    objectives: impl ObjectiveFunction,
    constraint: impl Constraint,
    callback: fn(usize, f32, f32),
) {
    let mut population_pool = PopulationPool::new(config.population_size, graph.vertices());
    initial_population(&mut population_pool, graph.vertices(), config.population_size, constraint, rng);
    let mut offspring_population = PopulationPool::new(config.population_size, graph.vertices());

    let max_iterations = config.max_iterations.unwrap_or(usize::max_value());

    for i in 0..max_iterations {
        let mut population: Vec<Specimen> = population_pool
            .chromosomes_mut()
            .map(|it| Specimen::from_chromosome(Chromosome::from_slice(it)))
            .collect();

        let mut offspring: Vec<Specimen> = offspring_population
            .chromosomes_mut()
            .map(|it| Specimen::from_chromosome(Chromosome::from_slice(it)))
            .collect();

        population.par_iter_mut().for_each(|specimen| {
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

        for offspring in offspring.iter_mut().take(config.population_size / 2) {
            let ancestor = tournament_succession(&pop1, config.tournament_size, |s| s.f1.unwrap().into(), rng);
            offspring.chromosome.clone_genes_from(&ancestor.chromosome);
            offspring.f1 = None;
            offspring.f2 = None;
        }

        for offspring in offspring.iter_mut().skip(config.population_size / 2) {
            let ancestor = tournament_succession(&pop2, config.tournament_size, |s| s.f2.unwrap().into(), rng);
            offspring.chromosome.clone_genes_from(&ancestor.chromosome);
            offspring.f1 = None;
            offspring.f2 = None;
        }

        offspring.shuffle(rng);

        for specimen in offspring.iter_mut() {
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

        for pair in offspring.chunks_exact_mut(2) {
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

        std::mem::swap(&mut population_pool, &mut offspring_population);
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
    population_storage: &mut PopulationPool,
    vertices: usize,
    population_size: usize,
    constraint: impl Constraint,
    rng: &mut impl Rng,
) {
    let mut created = 0;
    while created < population_size {
        let mut ch = Chromosome::from_slice(population_storage.nth_mut(created));
        for j in 0..vertices {
            ch.set(j, rng.gen::<bool>().into());
        }

        if constraint.is_satisfied(&ch) {
            created += 1;
        }
    }
}