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

#[derive(Clone)]
pub struct BinaryChromosome {
    genes: Vec<Gene>,
}

impl BinaryChromosome {
    pub fn with_length(length: usize) -> Self {
        BinaryChromosome {
            genes: vec![Gene::Zero; length]
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
}

impl Display for Chromosome {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for gene in &self.genes {
            write!(f, "{}", if *gene == Gene::Zero { '0' } else { '1' })?;
        }
        Ok(())
    }
}

pub type Chromosome = BinaryChromosome;