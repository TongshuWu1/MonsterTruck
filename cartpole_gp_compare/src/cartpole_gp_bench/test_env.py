# test_env.py
# Put next to cartpole_env.py (same folder).
# Run: python test_env.py
#
# Controls: A left, D right, R reset, Q/ESC quit

import numpy as np
import pygame

from cartpole_env import make_env


def main():
    pygame.init()
    W, H = 900, 520
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("CartPole Manual Control (A/D)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 28)

    # Use rgb_array and display the returned frames in THIS window
    env = make_env(render_mode="rgb_array", seed=0, start_down=True, edge_respawn=True, respawn_penalty=-2.0)
    obs, _ = env.reset(seed=0)

    key_a = False
    key_d = False
    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    key_a = True
                elif event.key == pygame.K_d:
                    key_d = True
                elif event.key == pygame.K_r:
                    obs, _ = env.reset()
                elif event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False

            elif event.type == pygame.KEYUP:
                if event.key == pygame.K_a:
                    key_a = False
                elif event.key == pygame.K_d:
                    key_d = False

        if key_a and not key_d:
            u = -1.0
        elif key_d and not key_a:
            u = +1.0
        else:
            u = 0.0

        obs, reward, terminated, truncated, info = env.step(np.array([u], dtype=np.float32))
        if terminated or truncated:
            obs, _ = env.reset()

        frame = env.render()  # (H, W, 3) uint8
        if frame is None:
            screen.fill((0, 0, 0))
        else:
            # pygame wants (W, H, 3)
            surf = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            surf = pygame.transform.smoothscale(surf, (W, H))
            screen.blit(surf, (0, 0))

        x, xdot, th, thdot = float(obs[0]), float(obs[1]), float(obs[2]), float(obs[3])
        hud1 = f"u={u:+.1f}  x={x:+.2f}  th={th:+.2f}  r={reward:+.2f}"
        hud2 = "A/D: push   R: reset   Q/ESC: quit"
        screen.blit(font.render(hud1, True, (20, 20, 20)), (12, 12))
        screen.blit(font.render(hud2, True, (20, 20, 20)), (12, 40))

        pygame.display.flip()
        clock.tick(50)  # match env dt=0.02

    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
