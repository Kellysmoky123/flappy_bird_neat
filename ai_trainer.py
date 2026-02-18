import neat
import os
import pickle
import pygame
from flappy_game import FlappyGame, Bird, FPS

def run_generation(genomes, config, render=True):
    """Run one generation of birds"""
    # Create birds for each genome
    birds = []
    for genome_id, genome in genomes:
        bird = Bird(genome, config)
        birds.append(bird)
        genome.fitness = 0
    
    game = FlappyGame(birds)
    game.game_started = True
    
    # Run until all birds are dead or timeout
    while not game.game_over and game.frame_count < 6000:
        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
        
        # AI Logic
        for bird in birds:
            if bird.alive:
                # Get inputs
                inputs = game.get_game_state(bird)
                # Get output
                output = bird.neural_net.activate(inputs)
                # Jump if output > 0.5
                if output[0] > 0.5:
                    bird.jump()
                
                # Fitness for survival
                bird.genome.fitness += 0.1
        
        game.step()
        
        if render:
            game.draw()
            game.clock.tick(FPS)
            
def run_neat():
    """Main NEAT training function"""
    # Load NEAT configuration
    local_dir = os.path.dirname(__file__)
    config_file = os.path.join(local_dir, 'neat_config.txt')
    
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Check if a checkpoint exists
    checkpoint_file = 'neat-checkpoint-'
    checkpoint_files = [f for f in os.listdir(local_dir) if f.startswith(checkpoint_file)]
    if checkpoint_files:
        latest_checkpoint = max(checkpoint_files, key=lambda x: int(x.split('-')[-1]))
        checkpoint_path = os.path.join(local_dir, latest_checkpoint)
        print(f"Resuming from checkpoint: {checkpoint_path}")
        population = neat.Checkpointer.restore_checkpoint(checkpoint_path)
        # Force update the fitness threshold because the checkpoint has the old config
        population.config.fitness_threshold = 100000.0
    else:
        # Start fresh
        print("Starting new training")
        population = neat.Population(config)
    
    # Add reporters for tracking progress
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    # Use local_dir for checkpoint saves so they always go to the script directory
    checkpoint_prefix = os.path.join(local_dir, 'neat-checkpoint-')
    population.add_reporter(neat.Checkpointer(10, filename_prefix=checkpoint_prefix))
    
    # Run training until fitness threshold is reached
    winner = population.run(lambda genomes, config: run_generation(genomes, config, render=False), None)
    
    # Save the best genome
    best_bird_path = os.path.join(local_dir, 'best_bird.pickle')
    with open(best_bird_path, 'wb') as f:
        pickle.dump(winner, f)
    
    print(f'\nBest genome:\n{winner}')

def test_best_genome():
    """Test the best genome from training"""
    local_dir = os.path.dirname(__file__)
    best_bird_path = os.path.join(local_dir, 'best_bird.pickle')
    
    with open(best_bird_path, 'rb') as f:
        genome = pickle.load(f)
    
    config_file = os.path.join(local_dir, 'neat_config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    
    # Run one generation with just the best bird, rendering enabled
    run_generation([(1, genome)], config, render=True)

if __name__ == '__main__':
    choice = input("Run NEAT training (t) or test best genome (b)? ").strip().lower()
    if choice == 'b':
        test_best_genome()
    else:
        run_neat()
