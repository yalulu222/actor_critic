import pygame
import numpy as np
from ..settings import EnvSettings

class GameRenderer:
    def __init__(self, config: EnvSettings):
        self.config = config
        self.screen = None
        self.clock = pygame.time.Clock()
        self.font = None
        
    def init_pygame(self):
        """Initialize pygame resources."""
        if not pygame.get_init():
            pygame.init()
            self.font = pygame.font.SysFont('Arial', 24)
            
    def create_surface(self, width: int, height: int):
        """Create a new pygame surface."""
        self.init_pygame()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Air Hockey")
        return self.screen
    
    def render(self, screen, state):
        """Render the current game state."""
        if screen is None:
            return
            
        # Clear the screen
        screen.fill(tuple(self.config.background_color))
        
        # Draw center line
        pygame.draw.line(
            screen,
            tuple(self.config.border_color),
            (self.config.width // 2, 0),
            (self.config.width // 2, self.config.height),
            2
        )
        
        # Draw goals
        self._draw_goals(screen)
        
        # Draw puck
        self._draw_puck(screen, state[:2])
        
        # Draw paddle
        self._draw_paddle(screen, state[2:4])
        
        # Draw score
        self._draw_debug_info(screen, state)
        
        pygame.display.flip()
    
    def _draw_goals(self, screen):
        """Draw the goals on each side."""
        goal_h = self.config.goal_height
        goal_y = (self.config.height - goal_h) // 2
        
        # Left goal
        pygame.draw.rect(
            screen,
            (200, 50, 50),  # Red tint for goals
            (0, goal_y, 10, goal_h),
            0
        )
        
        # Right goal
        pygame.draw.rect(
            screen,
            (50, 50, 200),  # Blue tint for goals
            (self.config.width - 10, goal_y, 10, goal_h),
            0
        )
        
        # Goal borders
        pygame.draw.rect(
            screen,
            tuple(self.config.border_color),
            (0, goal_y, 10, goal_h),
            2
        )
        pygame.draw.rect(
            screen,
            tuple(self.config.border_color),
            (self.config.width - 10, goal_y, 10, goal_h),
            2
        )
    
    def _draw_puck(self, screen, position):
        """Draw the puck at the given position."""
        pygame.draw.circle(
            screen,
            tuple(self.config.puck_color),
            (int(position[0]), int(position[1])),
            self.config.puck_radius
        )
    
    def _draw_paddle(self, screen, position):
        """Draw the paddle at the given position."""
        # Draw main paddle
        pygame.draw.circle(
            screen,
            tuple(self.config.paddle_color),
            (int(position[0]), int(position[1])),
            self.config.paddle_radius
        )
        
    def _draw_debug_info(self, screen, state):
        """Draw debug information on the screen."""
        if self.font is None:
            return
            
        # Draw velocity vector
        puck_pos = state[:2]
        puck_vel = state[4:6]
        end_pos = puck_pos + puck_vel * 0.1  # Scale down for visibility
        
        pygame.draw.line(
            screen,
            (255, 0, 0),  # Red vector
            (int(puck_pos[0]), int(puck_pos[1])),
            (int(end_pos[0]), int(end_pos[1])),
            2
        )
        
        # Draw speed text
        speed = np.linalg.norm(puck_vel)
        speed_text = self.font.render(f"Speed: {speed:.1f}", True, (0, 0, 0))
        screen.blit(speed_text, (10, 10))
    
    def get_rgb_array(self, state):
        """Get the current frame as an RGB array."""
        if self.screen is None:
            self.create_surface(self.config.width, self.config.height)
        
        self.render(self.screen, state)
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
        )
    
    def close(self):
        if pygame.get_init():
            try:
                pygame.display.flip()
            except pygame.error:
                pass
            self.clock.tick(self.config.fps)
            pygame.quit()
