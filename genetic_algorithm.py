import random
import numpy as np
from typing import List, Tuple, Callable
from grid import start_game, move_up, move_down, move_left, move_right, get_current_state, add_new_2
import copy

# Genetic Algorithm Parameters
POPULATION_SIZE = 50
INITIAL_GENOME_LENGTH = 5  # Starting length of genome
GENOME_LENGTH_INCREMENT = 5  # How much to increase genome length by
INCREMENT_EVERY = 10  # Generations between length increases
MUTATION_RATE = 0.1

# ELITISM_RATE determines what percentage of the top-performing individuals
# get carried over to the next generation unchanged.
# - Value between 0.0 and 1.0 (typically 0.1-0.3)
# - Higher values preserve good solutions but may reduce diversity
# - Lower values allow more exploration but may lose good solutions
ELITISM_RATE = 0.2

GENERATIONS = 50

# Movement mapping
MOVES = {
    0: move_up,
    1: move_down,
    2: move_left,
    3: move_right
}

class Individual:
    def __init__(self, genome_length: int = INITIAL_GENOME_LENGTH):
        self.genome = [random.randint(0, 3) for _ in range(genome_length)]
        self.fitness = 0
    
    def mutate(self):
        for i in range(len(self.genome)):
            if random.random() < MUTATION_RATE:
                self.genome[i] = random.randint(0, 3)
    
    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        crossover_point = random.randint(1, len(self.genome) - 1)
        child1 = Individual(len(self.genome))
        child1.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2 = Individual(len(self.genome))
        child2.genome = other.genome[:crossover_point] + self.genome[crossover_point:]
        return child1, child2

def evaluate_fitness(individual: Individual) -> int:
    """Simulate a game using the individual's genome and return the fitness score"""
    server_seed = "abcdefghijklmnopqrstvwxyz"
    client_seed = "1234567890"
    nonce = 1
    
    grid = start_game(server_seed, client_seed, nonce)
    total_score = 0
    move_count = 0
    
    for move_gene in individual.genome:
        move_func = MOVES[move_gene]
        new_grid, changed = move_func(copy.deepcopy(grid))
        
        if changed:
            nonce += 1
            grid = new_grid
            move_count += 1
            grid = add_new_2(grid, server_seed, client_seed, nonce)
            
            # Score components
            max_tile = max(max(row) for row in grid)
            empty_cells = sum(row.count(0) for row in grid)
            total_score += max_tile * 10 + empty_cells * 5
            
            state = get_current_state(grid)
            if state != 'GAME NOT OVER':
                break
    
    # Bonus for reaching higher tiles
    max_tile = max(max(row) for row in grid)
    total_score += max_tile * 100
    
    individual.fitness = total_score
    return total_score

def create_new_generation(population: List[Individual], genome_length: int) -> List[Individual]:
    """Create a new generation using selection, crossover and mutation"""
    population.sort(key=lambda x: x.fitness, reverse=True)
    new_population = []
    
    # ELITISM_RATE implementation:
    # The top ELITISM_RATE percentage of individuals are carried over unchanged
    elite_size = int(ELITISM_RATE * POPULATION_SIZE)
    new_population.extend(population[:elite_size])
    
    # Tournament selection and crossover for remaining spots
    while len(new_population) < POPULATION_SIZE:
        # Select parents through tournament selection
        tournament = random.sample(population, k=4)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        parent1, parent2 = tournament[0], tournament[1]
        
        # Create offspring through crossover
        child1, child2 = parent1.crossover(parent2)
        
        # Apply mutation
        child1.mutate()
        child2.mutate()
        
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)
    
    return new_population[:POPULATION_SIZE]

def run_genetic_algorithm():
    """Main function to run the genetic algorithm"""
    current_genome_length = INITIAL_GENOME_LENGTH
    population = [Individual(current_genome_length) for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        # Increase genome length every INCREMENT_EVERY generations
        if generation > 0 and generation % INCREMENT_EVERY == 0:
            current_genome_length += GENOME_LENGTH_INCREMENT
            print(f"Increasing genome length to {current_genome_length}")
            
            # Extend existing individuals' genomes with random moves
            for individual in population:
                individual.genome.extend(
                    [random.randint(0, 3) for _ in range(GENOME_LENGTH_INCREMENT)]
                )
        
        # Evaluate fitness
        for individual in population:
            evaluate_fitness(individual)
        
        # Create new generation
        population = create_new_generation(population, current_genome_length)
        
        # Print stats
        best_fitness = max(ind.fitness for ind in population)
        avg_fitness = sum(ind.fitness for ind in population) / POPULATION_SIZE
        print(f"Gen {generation + 1}: Best = {best_fitness}, Avg = {avg_fitness}, Genome Len = {current_genome_length}")
    
    return max(population, key=lambda x: x.fitness)

if __name__ == "__main__":
    best_individual = run_genetic_algorithm()
    print(f"\nBest individual achieved fitness: {best_individual.fitness}")
    print(f"Genome length: {len(best_individual.genome)}")
    print(f"Sample moves: {best_individual.genome[:10]}...")