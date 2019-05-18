use std::fmt::Display;
use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Gene {
    Zero = 0,
    One = 1,
}

impl From<bool> for Gene {
    fn from(v: bool) -> Self {
        match v {
            false => Gene::Zero,
            true => Gene::One,
        }
    }
}

pub struct BinaryChromosome<'s> {
    genes: &'s mut [Gene],
}

impl<'a> BinaryChromosome<'a> {
    pub fn from_slice<'b: 'a>(slice: &'a mut [Gene]) -> Self {
        Self {
            genes: slice
        }
    }

    #[inline]
    pub fn count_genes_by_value(&self) -> (usize, usize) {
        let ones = self.genes
            .iter()
            .filter(|it| **it == Gene::One)
            .count();
        (self.genes.len() - ones, ones)
    }

    #[inline]
    pub fn set(&mut self, n: usize, gene: Gene) {
        self.genes[n] = gene;
    }

    #[inline]
    pub fn get(&self, n: usize) -> &Gene {
        &self.genes[n]
    }

    #[inline]
    pub fn get_mut(&mut self, n: usize) -> &mut Gene {
        &mut self.genes[n]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.genes.len()
    }

    #[inline]
    pub fn swap(&mut self, n1: usize, n2: usize) {
        self.genes.swap(n1, n2);
    }

    #[inline]
    pub fn toggle(&mut self, n: usize) {
        self.set(n, match self.get(n) {
            Gene::Zero => Gene::One,
            Gene::One => Gene::Zero,
        });
    }

    #[inline]
    pub fn clone_genes_from(&mut self, other: &Self) {
        self.genes.copy_from_slice(other.genes);
    }
}

impl Display for Chromosome<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for gene in self.genes.iter() {
            write!(f, "{}", if *gene == Gene::Zero { '0' } else { '1' })?;
        }
        Ok(())
    }
}

pub type Chromosome<'s> = BinaryChromosome<'s>;