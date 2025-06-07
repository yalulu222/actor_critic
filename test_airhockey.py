import numpy as np
import pygame
import sys
from airhockey_gym.airhockey_env import AirHockeyEnv
from airhockey_gym.settings import EnvSettings

def main():
    # Initialize pygame
    pygame.init()

    try:
        # Initialize the environment
        config = EnvSettings()
        env = AirHockeyEnv(config=config, render_mode='human')

        # Reset the environment
        state, _ = env.reset()
        done = False
        reward = 0

        # Main game loop
        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:
                        state, _ = env.reset()
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            # Get keyboard input for paddle control
            keys = pygame.key.get_pressed()
            action = np.zeros(2, dtype=np.float32)

            if keys[pygame.K_LEFT]:
                action[0] = -1.0
            if keys[pygame.K_RIGHT]:
                action[0] = 1.0
            if keys[pygame.K_UP]:
                action[1] = -1.0
            if keys[pygame.K_DOWN]:
                action[1] = 1.0

            # Normalize diagonal movement
            if np.linalg.norm(action) > 0:
                action = action / np.linalg.norm(action)

            # Take a step in the environment
            state, r, terminated, truncated, _ = env.step(action)
            reward += r
            # Print debug info
            done = terminated or truncated
            if done:
                print(f"Game over! Reward: {reward:.2f} | Score: {env.score} | Steps: {env.current_step}")
                state, _ = env.reset()
                reward = 0

            # Cap the frame rate
            clock.tick(60)

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Cleaning up...")
        if 'env' in locals():
            env.close()
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    main()
