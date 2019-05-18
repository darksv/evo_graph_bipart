use crate::chromosome::Chromosome;
use rand::Rng;
use std::collections::HashSet;

pub struct Graph<'storage> {
    number_of_vertices: usize,
    storage: &'storage mut [u32],
}

impl<'storage> Graph<'storage> {
    pub fn from_slice(number_of_vertices: usize, storage: &mut [u32]) -> Graph {
        assert_eq!(number_of_vertices * number_of_vertices, storage.len());
        Graph {
            number_of_vertices,
            storage,
        }
    }

    #[inline]
    pub fn get_edge(&self, i: usize, j: usize) -> Option<u32> {
        assert!(i < self.number_of_vertices);
        assert!(j < self.number_of_vertices);
        unsafe { self.get_edge_unchecked(i, j) }
    }

    #[inline]
    unsafe fn get_edge_unchecked(&self, i: usize, j: usize) -> Option<u32> {
        let weight = *self.storage.get_unchecked(i * self.number_of_vertices + j);
        if weight == 0 {
            None
        } else {
            Some(weight)
        }
    }

    #[inline]
    pub fn set_edge(&mut self, i: usize, j: usize, weight: u32) {
        self.storage[i * self.number_of_vertices + j] = weight;
        self.storage[j * self.number_of_vertices + i] = weight;
    }

    #[inline]
    pub fn clear(&mut self) {
        self.storage.iter_mut().for_each(|it| *it = 0);
    }

    #[inline]
    pub fn vertices(&self) -> usize {
        self.number_of_vertices
    }

    #[inline]
    pub fn edges(&self) -> usize {
        let arcs = self.storage.iter().filter(|it| **it != 0).count();
        arcs / 2
    }

    pub fn iter_connecting<'g>(&'g self, c: &'g Chromosome) -> impl Iterator<Item=Edge> + 'g {
        self.iter_edges().filter_map(move |(i, j)| {
            let weight = unsafe { self.get_edge_unchecked(i, j)? };
            if c.get(i) == c.get(j) {
                return None;
            }
            Some(Edge { i, j, weight })
        })
    }

    fn iter_edges<'g>(&'g self) -> impl Iterator<Item=(usize, usize)> + 'g {
        // Traverse only elements above the main diagonal...
        (0..self.vertices())
            .flat_map(move |i| {
                (i + 1..self.vertices())
                    .map(move |j| (i, j))
            })
    }
}

#[derive(Clone, Copy)]
pub struct Edge {
    pub i: usize,
    pub j: usize,
    pub weight: u32,
}

fn is_connected(graph: &Graph) -> bool {
    let mut visited = HashSet::new();
    let mut remaining = vec![0];

    while let Some(current) = remaining.pop() {
        if !visited.insert(current) {
            continue;
        }

        for j in 0..graph.vertices() {
            if graph.get_edge(current, j).is_some() {
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
