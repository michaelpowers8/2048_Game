from pygame import init,QUIT,quit
from pygame.event import get
import pygame
from window import *
from tile import *
import grid

# Driver code
if __name__ == '__main__':
    
# calling start_game function
# to initialize the matrix
    server,client = "abcdefghijklmnopqrstvwxyz","1234567890"
    nonce:int = 1
    mat:list[list[int]] = grid.start_game(server,client,nonce)

    init()

    screen:Surface = load_screen()
    draw_grid(screen)
    write_seeds(screen,server,client)
    draw_all_tiles(screen,mat)

    # Game loop
    running = True
    # while running:
    #     for event in get():
    #         if event.type == QUIT:
    #             running = False

    while(running):
        # we have to move up
        # Handle key presses
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                changed = False  # Track if the board changed
                
                if event.key in (pygame.K_w, pygame.K_UP):
                    mat, changed = grid.move_up(mat)
                elif event.key in (pygame.K_s, pygame.K_DOWN):
                    mat, changed = grid.move_down(mat)
                elif event.key in (pygame.K_a, pygame.K_LEFT):
                    mat, changed = grid.move_left(mat)
                elif event.key in (pygame.K_d, pygame.K_RIGHT):
                    mat, changed = grid.move_right(mat)

                # Only add new number and increment nonce if board changed
                if changed:
                    nonce += 1
                    grid.add_new_2(mat, server, client, nonce)
                    status = grid.get_current_state(mat)
                    if status != 'GAME NOT OVER':
                        running = False

                # Redraw the board
                screen.fill((255, 255, 255))  # Clear screen
                draw_grid(screen)
                draw_all_tiles(screen, mat)
                pygame.display.flip()
            if event.type == QUIT:
                running = False
                break
    quit()
    