"""
Author: Anson Li
Created on: 26/6/2026
Purpose: script for easier data recording
Location: project_dir/EEG/data_record.py
"""

import pygame
#import constants for easier access to key events
from pygame.locals import *
import random
import string
import data_receive as BCI
import matplotlib.pyplot as plt

def main():
    #init fields
    ##game
    pygame.init()
    pygame.font.init()
    clock = pygame.time.Clock()

    win = pygame.display.set_mode((800,600))
    width, height = win.get_size()

    size = int(width*0.1)
    center = (width/2, height/2)
    my_font = pygame.font.SysFont('Comic Sans MS', size)

    show = ""
    count = 0
    show_text = False
    allow_input = True
    buffer = False
    buffer_time = 1
    show_time = 2
    cycle_start = 0
    buffer_start = 0
    ##random character list
    ###length = 0 to manual input
    length = 0
    arr = random.choices(string.ascii_letters + string.digits, k=length)

    ##game loop
    run = True
    win.fill((0,0,0))

    BCI.start(BID=-2, port=4, plot_domain="f")
    #start live plot in non blocking mode
    ani = BCI.plot_data(block=False)
    
    win.fill((0,0,0))
    #for logging, start_time set only when end
    #phase = start/end
    #type = buffer/cycle
    def log(phase, type, current_time, count, start_time=-1):
        if phase == "start":
            print(f"Starting {count} {type} at {current_time//1000}s")
        elif phase == "end" and start_time != -1:
            print(f"Finished {count} {type} at {current_time//1000}s in {(current_time-start_time)//1000}s")

    """
    Edit by Anson
    Date: 18/7/2026
    Changes: wrap entire game flow in try and finally, only set run=False inside
    """
    try:
        if arr:
            print(arr)
            allow_input = False
            # send start marker of show
            show = arr.pop(0)
            BCI.put_marker(show, phase="start")
            count += 1
            show_text = True

            cycle_start = pygame.time.get_ticks()
            log("start", "cycle", cycle_start, count)
            display_time = cycle_start + (show_time * 1000)

        while run:
            current_time = pygame.time.get_ticks()
            #send gui events so chart update without freezing
            plt.pause(0.001)
            for e in pygame.event.get():
                if e.type == QUIT or (e.type == KEYDOWN and e.key == K_ESCAPE):
                    run = False
                elif e.type == KEYDOWN and not arr:
                    #if key pressed is not enter, key is alphanumeric, input is allowed and arr is empty
                    if e.key != K_RETURN and e.unicode.isalnum() and allow_input:
                        show += e.unicode
                        print(f"\rKeyboard: {show: <20}", end='', flush=True)

                    elif e.key == K_BACKSPACE and allow_input:
                        show = show[:-1]
                        print(f"\rKeyboard: {show: <20}", end='', flush=True)

                    elif e.key in (K_RETURN, K_KP_ENTER) and show and allow_input:
                        print()
                        #send start marker of show
                        BCI.put_marker(show, phase="start")
                        allow_input = False
                        count += 1
                        show_text = True

                        cycle_start = current_time
                        log("start", "cycle", current_time=current_time, count=count)
                        display_time = current_time + show_time*1000

            if length != 0 and not arr and not show_text and not buffer:
                run = False

            #showing text but passed display_time(2s)            
            if show_text and current_time >= display_time:
                #send end marker of show
                BCI.put_marker(show, phase="end")
                print(show)
                log("end", "cycle",current_time=current_time, count=count, start_time=cycle_start)
                show_text = False
                buffer = True
                buffer_start = current_time
                display_time = current_time + buffer_time*1000
            
            elif buffer and current_time >= display_time:
                buffer = False
                allow_input = True
                show = ""
                log("end", "buffer", current_time=current_time, count=count, start_time=buffer_start)
                print("Allow input")
                if arr:
                    show = arr.pop(0)
                    BCI.put_marker(show, phase="start")
                    count += 1
                    show_text = True
                    cycle_start = current_time
                    log("start", "cycle", current_time, count)
                    display_time = current_time + (show_time * 1000)

            if show_text:
                text_surface = my_font.render(show, False, (255, 255, 255))
                textRect = text_surface.get_rect(center=center)
                win.blit(text_surface, textRect)
            else:
                win.fill((0, 0, 0))
                
            #update full display surface to screen
            pygame.display.flip()
            clock.tick(60)
    finally:
        BCI.end()
        pygame.quit()

if __name__ == "__main__":
    main()