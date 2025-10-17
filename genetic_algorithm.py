import sys
import copy
import json
import math
import pygame
import random
import hashlib
from datetime import datetime
from typing import List, Tuple
from tile import draw_all_tiles
from window import load_screen, draw_grid, write_seeds
from grid import start_game, move_up, move_down, move_left, move_right, get_current_state, add_new_2

# Genetic Algorithm Parameters
POPULATION_SIZE = 100
INITIAL_GENOME_LENGTH = 5
GENOME_LENGTH_INCREMENT = 5
INCREMENT_EVERY = 50
MUTATION_RATE = 0.15
GENERATIONS = 100_000
SERVER_SEED = "abcdefghijklmnopqrstvwxyz"
CLIENT_SEED = "1234567890"
DIVERSITY_THRESHOLD = 0.0  
CRITICAL_DIVERSITY_THRESHOLD = 0.01 

# ELITISM_RATE determines what percentage of the top-performing individuals
# get carried over to the next generation unchanged.
# - Value between 0.0 and 1.0 (typically 0.1-0.3)
# - Higher values preserve good solutions but may reduce diversity
# - Lower values allow more exploration but may lose good solutions
ELITISM_RATE = 0.1

# Movement mapping
MOVES = {
    0: move_up,
    1: move_down,
    2: move_left,
    3: move_right
}

MOVE_NAMES = {
    0: "UP",
    1: "DOWN",
    2: "LEFT",
    3: "RIGHT"
}

class Individual:
    def __init__(self, genome_length: int = INITIAL_GENOME_LENGTH):
        self.genome = [random.randint(0, 3) for _ in range(genome_length)]
        self.fitness = 0
        self.max_tile = 0
    
    def mutate(self, generation_number: int):
        mutation_rate = get_mutation_rate(generation_number)
        for i in range(1,6):
            if random.random() < mutation_rate:
                self.genome[i*-1] = random.randint(0, 3)
    
    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        crossover_point = random.randint(1, len(self.genome) - 1)
        child1 = Individual(len(self.genome))
        child1.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2 = Individual(len(self.genome))
        child2.genome = other.genome[:crossover_point] + self.genome[crossover_point:]
        return child1, child2
    
    def __hash__(self):
        genome_str:str = str(",".join([str(move) for move in self.genome]))
        fitness_str:str = str(self.fitness)
        max_tile_str:str = str(self.max_tile)
        full_str:str = f"{genome_str} : {fitness_str} : {max_tile_str}"
        full_bytes:bytes = full_str.encode()
        return hashlib.sha256(full_bytes).hexdigest()
    
    def __str__(self):
        return f"Genome Length: {len(self.genome):,.0f}\nFitness: {self.fitness:,.5f}\nMax Tile: {self.max_tile:,.0f}\n"
    
def get_mutation_rate(generation: int) -> float:
    initial_rate = 0.15  # 10%
    final_rate = 0.001    # 0.1%
    decay_generations = int(GENERATIONS*0.95)  # Linearly decay over 9,000 gens
    if generation >= decay_generations:
        return final_rate
    return initial_rate - (initial_rate - final_rate) * (generation / decay_generations)

def calculate_diversity(population: List[Individual]) -> float:
    """
    Calculate the diversity of the population using average entropy per gene position.
    Returns a value between 0 and log(4), where higher values indicate more diversity.
    """
    if not population or len(population) == 1:
        return 0.0
    
    genomes = [ind.genome for ind in population]
    n = len(genomes)
    genome_length = len(genomes[0])
    
    total_entropy = 0.0
    for i in range(genome_length):
        # Count frequency of each move (0-3) at position i
        freq = [0] * 4
        for genome in genomes:
            # print(len(genome))
            move = genome[i]
            freq[move] += 1
        
        # Calculate entropy for this position
        entropy = 0.0
        for count in freq:
            if count > 0:
                p = count / n
                entropy -= p * math.log(p)
        total_entropy += entropy
    
    # Average entropy across all positions
    average_entropy = total_entropy / genome_length
    return average_entropy

def evaluate_fitness(individual: Individual) -> int:
    """Simulate a game using the individual's genome and return the fitness score"""
    nonce = 1
    
    grid = start_game(SERVER_SEED, CLIENT_SEED, nonce)
    total_score = 0
    
    for move_gene in individual.genome:
        move_func = MOVES[move_gene]
        new_grid, changed = move_func(copy.deepcopy(grid))
        
        if changed:
            nonce += 1
            grid = new_grid
            grid = add_new_2(grid, SERVER_SEED, CLIENT_SEED, nonce)
            
            max_tile = max(max(row) for row in grid)
            individual.max_tile = max_tile
            empty_cells = sum(row.count(0) for row in grid)
            total_score += (max_tile * 10) + (empty_cells * 5)
            
            state = get_current_state(grid)
            if state != 'GAME NOT OVER':
                total_score *= 0.1
                break
        else:
            total_score *= 0.5 # 50% penalty for not moving the grid effectively wasting turns.
    
    max_tile = max(max(row) for row in grid)
    
    # Corner bonus (50% increase and 50% penalty if the max tile is one of the 4 center tiles) 
    corner_bonus = 1.0
    if max_tile in [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]:
        corner_bonus = 1.5
    elif max_tile in [grid[1][1], grid[1][2], grid[2][1], grid[2][2]]:
        corner_bonus = 0.5
    
    # Chain bonus calculation
    def calculate_chain_bonus(grid, max_tile):
        # Find position of max tile
        max_positions = [(i, j) for i in range(4) for j in range(4) if grid[i][j] == max_tile]
        if not max_positions:
            return 1.0
        
        total_bonus = 0.0
        current_value = max_tile
        
        for (i, j) in max_positions:
            # Check adjacent tiles for next values in the sequence
            neighbors = [
                (i-1, j), (i+1, j), (i, j-1), (i, j+1)  # Up, Down, Left, Right
            ]
            
            # Filter valid neighbors
            valid_neighbors = [
                (x, y) for (x, y) in neighbors 
                if 0 <= x < 4 and 0 <= y < 4 and grid[x][y] > 0
            ]
            
            # Look for chain patterns
            for x, y in valid_neighbors:
                if grid[x][y] == current_value // 2:  # Next in sequence (e.g., 1024 -> 512)
                    total_bonus += 0.2  # 20% bonus per link in chain
                    # Optional: Recursively check for longer chains
                    # total_bonus += 0.05 * calculate_chain_bonus(grid, current_value // 2)
        
        return 1.0 + min(total_bonus, 1.0)  # Cap chain bonus at 50%
    
    chain_bonus = calculate_chain_bonus(grid, max_tile)
    
    # Apply bonuses
    total_score += max_tile * 100 * corner_bonus * chain_bonus
    individual.fitness = total_score
    # if nonce < len(individual.genome)+1:
    #     individual.genome = individual.genome[:nonce]
    return total_score

def get_max_tile(genome:list[int]):
    """Simulate a game using the individual's genome and return the fitness score"""
    nonce = 1
    
    grid = start_game(SERVER_SEED, CLIENT_SEED, nonce)

    for move_gene in genome:
        move_func = MOVES[move_gene]
        new_grid, changed = move_func(copy.deepcopy(grid))
        
        if changed:
            nonce += 1
            grid = new_grid
            grid = add_new_2(grid, SERVER_SEED, CLIENT_SEED, nonce)
            
            max_tile = max(max(row) for row in grid)
            
            state = get_current_state(grid)
            if state != 'GAME NOT OVER':
                break
    
    max_tile = max(max(row) for row in grid)
    return max_tile

def create_new_generation(population: List[Individual], genome_length: int, generation_number: int) -> List[Individual]:
    population.sort(key=lambda x: x.fitness, reverse=True)
    new_population = []
    
    # Calculate current diversity
    current_diversity = calculate_diversity(population)
    
    elite_size = int(ELITISM_RATE * POPULATION_SIZE)
    new_population.extend(population[:elite_size])
    
    # Adjust selection pressure based on diversity
    tournament_size = 4  # Default
    if current_diversity < DIVERSITY_THRESHOLD:
        # Increase tournament size to reduce selection pressure when diversity is low
        tournament_size = 8
        print(f"Low diversity ({current_diversity:.3f}), increasing tournament size to {tournament_size}")
    
    while len(new_population) < POPULATION_SIZE:
        # Use adaptive tournament selection
        tournament = random.sample(population, k=tournament_size)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        parent1, parent2 = tournament[0], tournament[1]
        
        # child1, child2 = parent1.crossover(parent2)
        # child1.mutate(generation_number)
        # child2.mutate(generation_number)
        parent1.mutate(generation_number)
        parent2.mutate(generation_number)
        
        new_population.append(parent1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(parent2)
    
    # If diversity is critically low, introduce some random individuals
    # if current_diversity < CRITICAL_DIVERSITY_THRESHOLD:
    #     num_replace = int(POPULATION_SIZE * 0.1)  # Replace 10% of population
    #     for i in range(num_replace):
    #         # Replace worst individuals with new random ones
    #         new_population[-(i+1)] = Individual(genome_length)
    #     print(f"Critical diversity ({current_diversity:.3f}), replaced {num_replace} individuals")
    
    return new_population[:POPULATION_SIZE]

def run_genetic_algorithm():
    current_genome_length = INITIAL_GENOME_LENGTH
    population = [Individual(current_genome_length) for _ in range(POPULATION_SIZE)]
    previous_best_fitness_score = 0  # Track the best max tile from previous increments
    previous_average_fitness = 0
    
    for generation in range(GENERATIONS):
        # Evaluate fitness for all individuals
        for individual in population:
            evaluate_fitness(individual)
        
        # Find the best individual for this generation
        best_individual = max(population, key=lambda x: x.fitness)
        average_fitness = sum(ind.fitness for ind in population)/POPULATION_SIZE
        print(individual)
        
        # Check every INCREMENT_EVERY generations
        if generation > 0 and generation % INCREMENT_EVERY == 0:
            if((best_individual.fitness > previous_best_fitness_score) and (average_fitness>previous_average_fitness)):
                current_genome_length += GENOME_LENGTH_INCREMENT
                previous_best_fitness_score = best_individual.fitness
                previous_average_fitness = average_fitness
                for individual in population:
                    # Extend each individual's genome with random moves
                    individual.genome.extend([random.randint(0, 3) for _ in range(GENOME_LENGTH_INCREMENT)])
            #else:
                #print(f"Best Previous Fitness: {previous_best_fitness_score:,.2f}\nCurrent Best Fitness: {best_individual.fitness:,.2f}\nPrevious Average Fitness: {previous_average_fitness:,.2f}\nCurrent Average Fitness: {average_fitness:,.2f}")
        
        # Create new generation and print stats
        population = create_new_generation(population.copy(), current_genome_length, generation)
        
        best_fitness = max(ind.fitness for ind in population)
        best_fitness_individual = population[0]
        for current_individual in population:
            if current_individual.fitness > best_fitness_individual.fitness:
                best_fitness_individual = current_individual
        if((generation+1)%1_000==0):
            print(f"{datetime.now()} Gen {generation + 1}: Highest Tile = {get_max_tile(best_fitness_individual.genome)}, Best = {best_fitness}, Avg = {average_fitness}, Genome Len = {current_genome_length}, Mutation Rate: {get_mutation_rate(generation)*100}%")
    
    return max(population, key=lambda x: x.fitness)

def play_best_individual(best_individual: Individual):
    """Visualize the best individual playing the game with Pygame"""
    pygame.init()
    screen = load_screen()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
    
    nonce = 1
    
    grid = start_game(SERVER_SEED, CLIENT_SEED, nonce)
    move_index = 0
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                sys.exit()
        
        if move_index < len(best_individual.genome):
            move_gene = best_individual.genome[move_index]
            move_func = MOVES[move_gene]
            new_grid, changed = move_func(copy.deepcopy(grid))
            
            if changed:
                nonce += 1
                grid = new_grid
                grid = add_new_2(grid, SERVER_SEED, CLIENT_SEED, nonce)
                              
                # Draw everything
                screen.fill((255, 255, 255))
                draw_grid(screen)
                write_seeds(screen, SERVER_SEED, CLIENT_SEED)
                draw_all_tiles(screen, grid)
                
                # Display current move information
                move_text = font.render(f"Move {move_index}/{len(best_individual.genome)}: {MOVE_NAMES[move_gene]}", True, (0, 0, 0))
                screen.blit(move_text, (20, 650))
                
                pygame.display.flip()
                clock.tick(2)  # 2 moves per second
            
            state = get_current_state(grid)
            if state != 'GAME NOT OVER':
                result_text = font.render(f"Game Over: {state}", True, (255, 0, 0))
                screen.blit(result_text, (300, 650))
                pygame.display.flip()
                pygame.time.wait(3000)  # Show result for 3 seconds
                running = False
            move_index += 1
        else:
            running = False
        
    pygame.quit()

def play_previously_saved_individual():
    with open("Best_Sequence.json","r") as file:
        data:list[int] = json.load(file)
    individual_to_play:Individual = Individual(data["Number_of_Moves"])
    individual_to_play.genome = json.loads(data["Best_Sequence"])
    play_best_individual(individual_to_play)

def main():
    play_previous_best:bool = False
    if(play_previous_best):
        play_previously_saved_individual()
    else:
        best_individual = run_genetic_algorithm()
        written_genome:list[str] = []
        for best_move in best_individual.genome:
            written_genome.append(MOVE_NAMES[best_move])
        print(f"\nBest individual achieved fitness: {best_individual.fitness}")
        print(f"Genome length: {len(best_individual.genome)}")
        print("Launching Pygame visualization...")
        with open("Best_Sequence.json","w") as file:
            data:dict[str,str|int|float|list[int]] = {
                        "Best_Individual_Representation":repr(best_individual),
                        "Best_Individual_ID":id(best_individual),
                        "Best_Fitness":best_individual.fitness,
                        "Best_Sequence":f"[{','.join([str(x) for x in best_individual.genome])}]",
                        "Number_of_Generations":GENERATIONS,
                        "Number_of_Moves":len(best_individual.genome),
                        "Initial_Mutation_Rate":(MUTATION_RATE),
                        "End_Mutation_Rate":get_mutation_rate(GENERATIONS),
                        "Biggest_Tile_Reached":get_max_tile(best_individual.genome)
                    }
            json.dump(data,file,indent=4)

if __name__ == "__main__":
    main()