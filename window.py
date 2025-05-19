from pygame.display import set_mode,set_caption,flip
from pygame.draw import rect,line
from pygame.surface import Surface
from pygame.font import init,SysFont

def load_screen() -> Surface:
    # Set up the game window
    screen:Surface = set_mode((700, 700))
    set_caption("2048")
    screen.fill((255,255,255))
    return screen

def draw_grid(screen:Surface):
    rect(screen,(0,0,0),[99,99,503,503],4) # border

    line(screen,(0,0,0),[100,225],[598,225],3) # horizontal line top
    line(screen,(0,0,0),[100,350],[598,350],3) # horizontal line middle
    line(screen,(0,0,0),[100,475],[598,475],3) # horizontal line bottom

    line(screen,(0,0,0),[225,100],[225,598],3) # vertical line left
    line(screen,(0,0,0),[350,100],[350,598],3) # vertical line center
    line(screen,(0,0,0),[475,100],[475,598],3) # vertical line right
    flip() # Update screen

def write_seeds(screen:Surface,server:str,client:str) -> None:
    init()
    font1 = SysFont('freesanbold.ttf', 24)
    # Render the texts that you want to display
    text1 = font1.render(f"Server Seed: {server}        Client Seed: {client}", True, (0, 0, 0))
    # create a rectangular object for the
    # text surface object
    textRect1 = text1.get_rect()
    # setting center for the first text
    textRect1.center = (350, 50)
    screen.blit(text1,textRect1)
    flip()