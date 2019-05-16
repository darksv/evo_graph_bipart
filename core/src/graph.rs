use crate::chromosome::Chromosome;

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
    pub fn get_edge(&self, i: usize, j: usize) -> u32 {
        self.storage[i * self.number_of_vertices + j]
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

    pub fn iter_connecting<'g>(&'g self, c: &'g Chromosome) -> impl Iterator<Item=(usize, usize)> + 'g {
        self.iter_edges().filter(move |&(i, j)| self.get_edge(i, j) != 0 && c.get(i) != c.get(j))
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