import random
import pygame
import sys
from typing import List, Tuple
from grid import start_game, move_up, move_down, move_left, move_right, get_current_state, add_new_2
from window import load_screen, draw_grid, write_seeds
from tile import draw_all_tiles
import copy
import json

# Genetic Algorithm Parameters
POPULATION_SIZE = 10
INITIAL_GENOME_LENGTH = 5
GENOME_LENGTH_INCREMENT = 5
INCREMENT_EVERY = 1
MUTATION_RATE = 0.15
GENERATIONS = 100

# ELITISM_RATE determines what percentage of the top-performing individuals
# get carried over to the next generation unchanged.
# - Value between 0.0 and 1.0 (typically 0.1-0.3)
# - Higher values preserve good solutions but may reduce diversity
# - Lower values allow more exploration but may lose good solutions
ELITISM_RATE = 0.3

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
    
    def mutate(self,generation_number:int):
        for i in range(len(self.genome)):
            if random.random() < get_mutation_rate(generation_number):
                self.genome[i] = random.randint(0, 3)
    
    def crossover(self, other: 'Individual') -> Tuple['Individual', 'Individual']:
        crossover_point = random.randint(1, len(self.genome) - 1)
        child1 = Individual(len(self.genome))
        child1.genome = self.genome[:crossover_point] + other.genome[crossover_point:]
        child2 = Individual(len(self.genome))
        child2.genome = other.genome[:crossover_point] + self.genome[crossover_point:]
        return child1, child2
    
def get_mutation_rate(generation: int) -> float:
    initial_rate = 0.15  # 10%
    final_rate = 0.001    # 1%
    decay_generations = 7500  # Linearly decay over 5,000 gens
    if generation >= decay_generations:
        return final_rate
    return initial_rate - (initial_rate - final_rate) * (generation / decay_generations)

def evaluate_fitness(individual: Individual) -> int:
    """Simulate a game using the individual's genome and return the fitness score"""
    server_seed = "abcdefghijklmnopqrstvwxyz"
    client_seed = "1234567890"
    nonce = 1
    
    grid = start_game(server_seed, client_seed, nonce)
    total_score = 0
    
    for move_gene in individual.genome:
        move_func = MOVES[move_gene]
        new_grid, changed = move_func(copy.deepcopy(grid))
        
        if changed:
            nonce += 1
            grid = new_grid
            grid = add_new_2(grid, server_seed, client_seed, nonce)
            
            max_tile = max(max(row) for row in grid)
            empty_cells = sum(row.count(0) for row in grid)
            total_score += max_tile * 10 + empty_cells * 5
            
            state = get_current_state(grid)
            if state != 'GAME NOT OVER':
                total_score *= 0.3
                break
    
    max_tile = max(max(row) for row in grid)
    
    # Corner bonus (50% increase and 20% penalty if the max tile is one of the 4 center tiles) 
    corner_bonus = 1.5 if max_tile in [grid[0][0], grid[0][-1], grid[-1][0], grid[-1][-1]] else 1.0
    corner_bonus = 0.8 if max_tile in [grid[1][1], grid[1][2], grid[2][1], grid[2][2]] else 1.0
    
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
                    total_bonus += 0.1  # 10% bonus per link in chain
                    # Optional: Recursively check for longer chains
                    # total_bonus += 0.05 * calculate_chain_bonus(grid, current_value // 2)
        
        return 1.0 + min(total_bonus, 0.5)  # Cap chain bonus at 50%
    
    chain_bonus = calculate_chain_bonus(grid, max_tile)
    
    # Apply bonuses
    total_score += max_tile * 100 * corner_bonus * chain_bonus
    individual.fitness = total_score
    return total_score

def get_max_tile(genome:list[int]):
    """Simulate a game using the individual's genome and return the fitness score"""
    server_seed = "abcdefghijklmnopqrstvwxyz"
    client_seed = "1234567890"
    nonce = 1
    
    grid = start_game(server_seed, client_seed, nonce)

    for move_gene in genome:
        move_func = MOVES[move_gene]
        new_grid, changed = move_func(copy.deepcopy(grid))
        
        if changed:
            nonce += 1
            grid = new_grid
            grid = add_new_2(grid, server_seed, client_seed, nonce)
            
            max_tile = max(max(row) for row in grid)
            
            state = get_current_state(grid)
            if state != 'GAME NOT OVER':
                break
    
    max_tile = max(max(row) for row in grid)
    return max_tile

def create_new_generation(population: List[Individual], genome_length: int, generation_number:int) -> List[Individual]:
    population.sort(key=lambda x: x.fitness, reverse=True)
    new_population = []
    
    elite_size = int(ELITISM_RATE * POPULATION_SIZE)
    new_population.extend(population[:elite_size])
    
    while len(new_population) < POPULATION_SIZE:
        tournament = random.sample(population, k=4)
        tournament.sort(key=lambda x: x.fitness, reverse=True)
        parent1, parent2 = tournament[0], tournament[1]
        
        child1, child2 = parent1.crossover(parent2)
        child1.mutate(generation_number)
        child2.mutate(generation_number)
        
        new_population.append(child1)
        if len(new_population) < POPULATION_SIZE:
            new_population.append(child2)
    
    return new_population[:POPULATION_SIZE]

def run_genetic_algorithm():
    current_genome_length = INITIAL_GENOME_LENGTH
    population = [Individual(current_genome_length) for _ in range(POPULATION_SIZE)]
    
    for generation in range(GENERATIONS):
        if generation > 0 and generation % INCREMENT_EVERY == 0:
            current_genome_length += GENOME_LENGTH_INCREMENT
            #print(f"Increasing genome length to {current_genome_length}")
            for individual in population:
                individual.genome.extend([random.randint(0, 3) for _ in range(GENOME_LENGTH_INCREMENT)])
        
        for individual in population:
            evaluate_fitness(individual)
        
        population = create_new_generation(population, current_genome_length,generation)
        
        best_fitness = max(ind.fitness for ind in population)
        avg_fitness = sum(ind.fitness for ind in population) / POPULATION_SIZE
        if((generation+1)%100==0):
            print(f"Gen {generation + 1}: Best = {best_fitness}, Avg = {avg_fitness}, Genome Len = {current_genome_length}, Mutation Rate: {get_mutation_rate(generation)*100}%")
    
    return max(population, key=lambda x: x.fitness)

def play_best_individual(best_individual: Individual):
    """Visualize the best individual playing the game with Pygame"""
    pygame.init()
    screen = load_screen()
    clock = pygame.time.Clock()
    font = pygame.font.SysFont('Arial', 24)
    
    server_seed = "abcdefghijklmnopqrstvwxyz"
    client_seed = "1234567890"
    nonce = 1
    
    grid = start_game(server_seed, client_seed, nonce)
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
                grid = add_new_2(grid, server_seed, client_seed, nonce)
                              
                # Draw everything
                screen.fill((255, 255, 255))
                draw_grid(screen)
                write_seeds(screen, server_seed, client_seed)
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
    individual_to_play:Individual = Individual(len(data))
    individual_to_play.genome = data.copy()
    play_best_individual(individual_to_play)

if __name__ == "__main__":
    if(True):
        play_previously_saved_individual()
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
    while True:
        play_best_individual(best_individual)