from pygame.display import flip
from pygame.draw import rect
from pygame.surface import Surface
from pygame.font import init,SysFont

def draw_new_tile(screen:Surface,position:tuple[int,int],value:int):
    rect(screen,(186,172,151),[position[0],position[1],122,122],0)
    # Create a font file by passing font file
    # and size of the font
    init()
    font1 = SysFont('freesanbold.ttf', 50)
    # Render the texts that you want to display
    text1 = font1.render(str(value), True, (0, 0, 0))
    # create a rectangular object for the
    # text surface object
    textRect1 = text1.get_rect()
    # setting center for the first text
    textRect1.center = (position[0]+61, position[1]+61)
    screen.blit(text1,textRect1)
    flip()

def draw_all_tiles(screen:Surface,grid:list[list[int]]):
    for i,row in enumerate(grid):
        for j,number in enumerate(row):
            if(number==0):
                pass
            else:
                draw_new_tile(screen,(102+(125*j),102+(125*i)),number)