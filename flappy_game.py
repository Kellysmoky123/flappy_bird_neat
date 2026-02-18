import pygame
import random
import neat

# Initialize Pygame
pygame.init()

# Game Constants
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (135, 206, 235)
YELLOW = (255, 255, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
DARK_GREEN = (0, 128, 0)

# Bird constants
BIRD_WIDTH = 30
BIRD_HEIGHT = 25
BIRD_X = 80  # Fixed position for bird
GRAVITY = 0.5
JUMP_STRENGTH = -6  # Negative because y is 0 at top

# Pipe Constants
PIPE_WIDTH = 50
PIPE_GAP = 150  # Gap between top and bottom pipe
PIPE_SPEED = 3  # How fast pipe move left


class Bird:
    def __init__(self, genome=None, config=None):
        self.x = BIRD_X
        self.y = SCREEN_HEIGHT // 2  # start in the middle of the screen
        self.width = BIRD_WIDTH
        self.height = BIRD_HEIGHT
        self.velocity = 0
        self.angle = 0
        self.alive = True
        self.score = 0
        self.time_alive = 0

        # ai related attributes
        self.genome = genome
        self.neural_net = None
        self.fitness = 0
        self.distance_travelled = 0

        # If we have genome, create neural net
        if genome and config:
            self.neural_net = neat.nn.FeedForwardNetwork.create(genome, config)

    def update(self):
        if not self.alive:
            return
        # Apply gravity - bird falls faster over time
        self.velocity += GRAVITY
        self.y += self.velocity

        # Increment time alive for fitness calculation
        self.time_alive += 1

        # Update angle based on velocity
        self.angle = min(max(self.velocity * 3, -45), 90)

        # Check boundaries - kill bird if it hits floor or ceiling
        if self.y <= 0 or self.y >= SCREEN_HEIGHT - self.height:
            # Clamp position before marking dead
            self.y = max(0, min(self.y, SCREEN_HEIGHT - self.height))
            self.velocity = 0
            self.alive = False

    def jump(self):
        if self.alive:
            self.velocity = JUMP_STRENGTH  # Gives bird upward velocity

    def draw(self, screen):
        # Draw bird as a yellow circle
        if not self.alive:
            return
        center_x = int(self.x + self.width // 2)
        center_y = int(self.y + self.height // 2)
        radius = self.width // 2

        pygame.draw.circle(screen, YELLOW, (center_x, center_y), radius)
        pygame.draw.circle(screen, BLACK, (center_x, center_y), radius, 2)

        # Add a simple eye
        eye_x = center_x + 5
        eye_y = center_y + 3
        pygame.draw.circle(screen, WHITE, (eye_x, eye_y), 5)
        pygame.draw.circle(screen, BLACK, (eye_x, eye_y), 3)
        
        # Draw a simple beak
        beak_points = [(center_x + radius, center_y),
                      (center_x + radius + 8, center_y - 2),
                      (center_x + radius, center_y + 2)]
        pygame.draw.polygon(screen, (255, 165, 0), beak_points)

    def get_rect(self):
        return pygame.Rect(self.x, self.y, self.width, self.height)


class Pipe:
    def __init__(self, x):
        self.x = x
        self.width = PIPE_WIDTH

        # Create a random gap position somewhere in the middle area of screen
        min_gap_y = 100  # Don't put gap too high
        max_gap_y = SCREEN_HEIGHT - PIPE_GAP - 100  # Don't put gap too low
        self.gap_y = random.randint(min_gap_y, max_gap_y)
        self.scored = False  # Whether the game score was incremented for this pipe
        self.passed_birds = set()  # Track which birds have been rewarded

        # Calculate pipe heights
        self.top_height = self.gap_y  # top pipe from top to gap_y
        self.bottom_y = self.gap_y + PIPE_GAP  # bottom pipe starts after gap
        self.bottom_height = (
            SCREEN_HEIGHT - self.bottom_y
        )  # bottom pipe to screen bottom

    def draw(self, screen):
        # draw top pipe
        top_rect = pygame.Rect(self.x, 0, self.width, self.top_height)
        pygame.draw.rect(screen, GREEN, top_rect)
        pygame.draw.rect(screen, DARK_GREEN, (top_rect.x, top_rect.y, 5, top_rect.height))
        pygame.draw.rect(screen, BLACK, top_rect, 2)

        # bottom pipe
        bottom_rect = pygame.Rect(self.x, self.bottom_y, self.width, self.bottom_height)
        pygame.draw.rect(screen, GREEN, bottom_rect)
        pygame.draw.rect(screen, DARK_GREEN, (bottom_rect.x, bottom_rect.y, 5, bottom_rect.height))
        pygame.draw.rect(screen, BLACK, bottom_rect, 2)

        # Add pipe cap for realistic look
        cap_height = 30
        cap_width = self.width + 6

        #top cap
        top_cap = pygame.Rect(self.x - 3, self.top_height, cap_width, cap_height)
        pygame.draw.rect(screen, GREEN, top_cap)
        pygame.draw.rect(screen, BLACK, top_cap, 2)

        # bottom cap
        bottom_cap = pygame.Rect(self.x - 3, self.bottom_y, cap_width, cap_height)
        pygame.draw.rect(screen, GREEN, bottom_cap)
        pygame.draw.rect(screen, BLACK, bottom_cap, 2)

    def update(self):
        # Move pipe to the left
        self.x -= PIPE_SPEED

    def is_off_screen(self):
        # Check if pipe is off screen
        return self.x + self.width < 0

    def get_top_rect(self):
        # Return rectangle for top pipe collision
        return pygame.Rect(self.x, 0, self.width, self.top_height)

    def get_bottom_rect(self):
        # Return rectangle for bottom pipe collision
        return pygame.Rect(self.x, self.bottom_y, self.width, self.bottom_height)

    def collides_with(self, bird):
        if not bird.alive:
            return False
        bird_rect = bird.get_rect()
        top_rect = self.get_top_rect()
        bottom_rect = self.get_bottom_rect()

        cap_height = 30
        cap_width = self.width + 6
    
        top_cap = pygame.Rect(self.x - 3, self.top_height, cap_width, cap_height)
        bottom_cap = pygame.Rect(self.x - 3, self.bottom_y, cap_width, cap_height)

        return bird_rect.colliderect(top_rect) or bird_rect.colliderect(bottom_rect) or bird_rect.colliderect(top_cap) or bird_rect.colliderect(bottom_cap)

    def bird_passed(self, bird):
        """Check if a specific bird has passed this pipe (for fitness rewards)."""
        return bird.alive and (id(bird) not in self.passed_birds) and (bird.x > self.x + self.width)

def draw_background(screen):
    # Draw a gradient sky background
    for y in range(SCREEN_HEIGHT):
        # Create gradient from light blue to darker blue
        color_intensity = int(135 + (206 - 135) * (y / SCREEN_HEIGHT))
        color = (color_intensity, 206, 235)
        pygame.draw.line(screen, color, (0, y), (SCREEN_WIDTH, y))


class FlappyGame:
    def __init__(self, birds=None):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Flappy Bird")
        self.clock = pygame.time.Clock()
        
        # If no birds provided, create one for human player
        if birds is None:
            self.birds = [Bird()]
            self.human_mode = True
        else:
            self.birds = birds
            self.human_mode = False
            
        self.pipes = []
        self.pipe_timer = 0
        self.pipe_spawn_delay = 90
        self.score = 0
        self.high_score = 0
        self.game_over = False
        self.game_started = False
        self.frame_count = 0

    def reset(self):
        if self.score > self.high_score:
            self.high_score = self.score
        
        if self.human_mode:
            self.birds = [Bird()]
        
        self.pipes = []
        self.game_over = False
        self.game_started = False
        self.score = 0
        self.pipe_timer = 0
        self.frame_count = 0

    def get_game_state(self, bird):
        """Extract game state for neural network input"""
        # Find the next pipe the bird needs to pass
        next_pipe = None
        for pipe in self.pipes:
            if pipe.x + pipe.width > bird.x:
                next_pipe = pipe
                break
        
        if next_pipe:
            # Distance to next pipe
            horizontal_distance = next_pipe.x - bird.x
            # Distance to top and bottom of gap
            vertical_distance_top = bird.y - next_pipe.gap_y
            vertical_distance_bottom = (next_pipe.gap_y + PIPE_GAP) - bird.y
            # Bird's velocity and position
            bird_velocity = bird.velocity
            bird_y = bird.y
        else:
            # No pipes ahead
            horizontal_distance = SCREEN_WIDTH
            vertical_distance_top = 0
            vertical_distance_bottom = 0
            bird_velocity = bird.velocity
            bird_y = bird.y
        
        # Normalize inputs
        inputs = [
            horizontal_distance / SCREEN_WIDTH,
            vertical_distance_top / SCREEN_HEIGHT,
            vertical_distance_bottom / SCREEN_HEIGHT,
            bird_velocity / 10,
            bird_y / SCREEN_HEIGHT
        ]
        return inputs

    def step(self):
        """Advance game physics by one frame"""
        self.frame_count += 1
        
        # Spawn pipes
        if self.game_started:
            self.pipe_timer += 1
            if self.pipe_timer >= self.pipe_spawn_delay:
                self.pipes.append(Pipe(SCREEN_WIDTH))
                self.pipe_timer = 0

        # Update birds
        for bird in self.birds:
            if bird.alive:
                bird.update()

                # Check pipe collisions (boundary death is handled in Bird.update)
                if bird.alive:  # Bird may have died from boundary in update()
                    for pipe in self.pipes:
                        if pipe.collides_with(bird):
                            bird.alive = False
                            if not self.human_mode:
                                bird.genome.fitness -= 5
                            break
                elif not self.human_mode:
                    # Bird died from boundary check
                    bird.genome.fitness -= 5

        # Update pipes
        for pipe in self.pipes[:]:
            pipe.update()

            # Check if any bird passed this pipe
            for bird in self.birds:
                if pipe.bird_passed(bird):
                    # Track this bird as having passed
                    pipe.passed_birds.add(id(bird))

                    # Increment game score once per pipe
                    if not pipe.scored:
                        pipe.scored = True
                        self.score += 1

                    # Reward each bird individually
                    if not self.human_mode:
                        bird.score += 1
                        bird.genome.fitness += 10

            if pipe.is_off_screen():
                self.pipes.remove(pipe)

        # Check game over
        if not any(b.alive for b in self.birds):
            self.game_over = True

    def draw(self):
        draw_background(self.screen)
        
        for pipe in self.pipes:
            pipe.draw(self.screen)
            
        for bird in self.birds:
            if bird.alive:
                bird.draw(self.screen)
                
        # Draw UI
        font = pygame.font.Font(None, 36)
        small_font = pygame.font.Font(None, 24)
        
        if not self.game_started and not self.game_over:
            # Start screen
            title_font = pygame.font.Font(None, 48)
            title_text = title_font.render("Flappy Bird", True, BLACK)
            start_text = small_font.render("Press SPACE to Start", True, BLACK)
            
            title_rect = title_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 100))
            start_rect = start_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
            
            self.screen.blit(title_text, title_rect)
            self.screen.blit(start_text, start_rect)
            
        elif self.game_started and not self.game_over:
            # Score
            score_text = font.render(f"Score: {self.score}", True, BLACK)
            self.screen.blit(score_text, (10, 10))
            
            if self.human_mode:
                instruction_text = small_font.render("Press SPACE to jump!", True, BLACK)
                self.screen.blit(instruction_text, (10, SCREEN_HEIGHT - 30))
            else:
                # AI Stats
                alive_count = len([b for b in self.birds if b.alive])
                alive_text = font.render(f"Alive: {alive_count}", True, BLACK)
                self.screen.blit(alive_text, (10, 50))
                
        elif self.game_over:
            game_over_text = font.render("GAME OVER", True, RED)
            final_score_text = small_font.render(f"Final Score: {self.score}", True, BLACK)
            restart_text = small_font.render("Press R to Restart", True, BLACK)
            
            game_over_rect = game_over_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50))
            final_score_rect = final_score_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 20))
            restart_rect = restart_text.get_rect(center=(SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 10))
            
            self.screen.blit(game_over_text, game_over_rect)
            self.screen.blit(final_score_text, final_score_rect)
            self.screen.blit(restart_text, restart_rect)

        pygame.display.flip()

def main():
    game = FlappyGame()
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not game.game_started and not game.game_over:
                        game.game_started = True
                    elif game.game_started and not game.game_over:
                        game.birds[0].jump()
                elif event.key == pygame.K_r and game.game_over:
                    game.reset()
        
        if game.game_started and not game.game_over:
            game.step()
            
        game.draw()
        game.clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()
