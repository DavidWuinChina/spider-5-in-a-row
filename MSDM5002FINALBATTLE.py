# 如果你完全按照我的指示进行了操作，那么这段代码可以正确运行
# 将“module_path”更改为您的文件名或文件夹名
# 注意
# 1）MSDM5002FINALSTRATEGYbigBOSS.py用于检测胜利的check_winner函数放在了MCTS类中
# 2）用于AI对战的MSDM5002FINALBATTLE.py中已将check_winner函数放在py中可以直接调用，不需使用"gp1.check_winner"或"gp2.check_winner",直接使用"check_winner(board)"即可
# 3）这段代码已将本团队的策略设置为gp1，与其他AI策略对战，只需修改"import MSDM5002FINALSAMPLE as gp2"中"MSDM5002FINALSAMPLE"即可
import sys
import MSDM5002FINALSTRATEGYbigBOSS as gp1
import pygame as pg
import numpy as np
import MSDM5002FINALSAMPLE as gp2
module_path = 'C:/ddm课件/科学编程与可视化5002/final project/pythonProject'
sys.path.append(module_path)
import importlib
importlib.reload(gp1)




####################################################################################################################
# create the initial empty chess board in the game window
# Game Configuration
W_SIZE = 720  # Window size
PAD = 36  # Padding size
RADIAL_SPAN = 10  # Number of radial circles
ANGULAR_SPAN = 16  # Number of angular lines
# To record the filling status of each point in the first column, initially set to False meaning all are unfilled
FIRST_COLUMN_FILLED = [False] * ANGULAR_SPAN
# Initialize Pygame
pg.init()
# Colors
COLOR_LINE = [153, 153, 153]
COLOR_BOARD = [241, 196, 15]
COLOR_BLACK = [0, 0, 0]
COLOR_WHITE = [255, 255, 255]
# Global variables
CENTER = W_SIZE / 2
SEP_R = 0
SEP_TH = 0
PIECE_RADIUS = 0
# Draw the board
def draw_board():
    global CENTER, SEP_R, SEP_TH, PIECE_RADIUS
    CENTER = W_SIZE / 2
    SEP_R = int((CENTER - PAD) / (RADIAL_SPAN - 1))  # Radial separation
    SEP_TH = 2 * np.pi / ANGULAR_SPAN  # Angular separation
    PIECE_RADIUS = SEP_R / 2 * SEP_TH * 0.8  # Piece size
    surface = pg.display.set_mode((W_SIZE, W_SIZE))  # Set window size
    pg.display.set_caption("Gomoku (Five-in-a-Row)")  # Set window title
    surface.fill(COLOR_BOARD)
    # Draw circles
    for i in range(1, RADIAL_SPAN):
        pg.draw.circle(surface, COLOR_LINE, (CENTER, CENTER), SEP_R * i, 3)
    # Draw radial lines
    for i in range(ANGULAR_SPAN // 2):
        pg.draw.line(surface, COLOR_LINE,
                     (CENTER + (CENTER - PAD) * np.cos(SEP_TH * i), CENTER + (CENTER - PAD) * np.sin(SEP_TH * i)),
                     (CENTER - (CENTER - PAD) * np.cos(SEP_TH * i), CENTER - (CENTER - PAD) * np.sin(SEP_TH * i)), 3)
    pg.display.update()
    return surface


####################################################################################################################
# translate clicking position on the window to array indices (th, r)
# pos = (x,y) is a tuple returned by pygame, telling where an event (i.e. player click) occurs on the game window
def click2index(pos):
    print(pos)
    dist = np.sqrt((pos[0] - CENTER) ** 2 + (pos[1] - CENTER) ** 2)
    print("dist", dist)
    print("th:",round(np.arctan2((pos[1] - CENTER), (pos[0] - CENTER)) / SEP_TH))
    print("r:", round(dist / SEP_R))
    if dist < W_SIZE / 2 - PAD + 0.25 * SEP_R:
        th, r = round(np.arctan2((pos[1] - CENTER), (pos[0] - CENTER)) / SEP_TH), round(dist / SEP_R)
        if th>=0:
            th=th
        else:
            th=th+16
        if r == 0 and FIRST_COLUMN_FILLED[th]==False:
            FIRST_COLUMN_FILLED[th]=True
            print("有一个为r=0")
            return (th,0)
        else:
            return (th,r)
    else:
        print("需要下在棋盘内")

####################################################################################################################
# Draw the stones on the board at pos = [th, r]
# r and th are the indices on the 16x10 board array (under rectangular grid representation)
# Draw a black circle at pos if color = 1, and white circle at pos if color =  -1

def draw_stone(surface, pos, color=0):
    x = CENTER + pos[1] * SEP_R * np.cos(pos[0] * SEP_TH)
    y = CENTER + pos[1] * SEP_R * np.sin(pos[0] * SEP_TH)
    if color == 1:
        pg.draw.circle(surface, COLOR_BLACK, (x, y), PIECE_RADIUS * (1 + 2 * pos[1] / RADIAL_SPAN), 0)
    elif color == -1:
        pg.draw.circle(surface, COLOR_WHITE, (x, y), PIECE_RADIUS * (1 + 2 * pos[1] / RADIAL_SPAN), 0)
    pg.display.update()


####################################################################################################################
def print_winner(surface, winner=0):
    if winner == 2:
        msg = "Draw! White wins"
        COLOR = [153, 153, 153]
    elif winner == 1:
        msg = "Black wins!"
        COLOR = [0, 0, 0]
    elif winner == -1:
        msg = "White wins!"
        COLOR = [255, 255, 255]
    else:
        return
    font = pg.font.Font('freesansbold.ttf', 32)
    text = font.render(msg, True, COLOR)
    textRect = text.get_rect()
    textRect.topleft = (0, 0)
    surface.blit(text, textRect)
    pg.display.update()




####################################################################################################################

def battle(ai_move1, ai_move2):
    def check_sequence(board, start_th, start_r, delta_th, delta_r, color):
        first = False
        if delta_th == 1 and delta_r == 0 and start_r == 0:
            for i in range(ANGULAR_SPAN):
                board[i][0] = color
            first = True
        if first:
            return
        for i in range(5):
            th = (start_th + delta_th * i) % ANGULAR_SPAN
            r = start_r + delta_r * i
            if r <= 0 or r >= RADIAL_SPAN or board[th][r] != color:
                return False
        return True

    def check_winner(board):
        directions = [
            (0, 1),  # Radial
            (1, 0),  # Circular
            (1, 1),  # Diagonal increasing theta and r
            (1, -1)  # Diagonal increasing theta, decreasing r
        ]

        for th in range(ANGULAR_SPAN):
            for r in range(RADIAL_SPAN):
                if board[th][r] == 0:
                    continue
                color = board[th][r]
                for delta_th, delta_r in directions:
                    if check_sequence(board, th, r, delta_th, delta_r, color):
                        return color  # Return the winner's color
            # Special win condition involving the center point (0, 0)
        for x2 in range(ANGULAR_SPAN):  # Iterate over all possible x2 values
            x1 = (x2 + 8) % ANGULAR_SPAN  # Calculate x1 based on the condition (16 - x1 + x2 = 8)
            for color in [1, -1]:  # Check for both black (1) and white (-1)
                # Check if the center point and adjacent cells form a valid combination
                for th in range(ANGULAR_SPAN):
                    if board[th][0] == color:  # Center point and any position with y-coordinate 0 must match the color
                        # Case 1: a=2, b=2 (x1, x2 both have 2 different radial cells)
                        if (board[x1][1] == color and board[x1][2] == color and
                                board[x2][1] == color and board[x2][2] == color):
                            return color
                        # Case 2: a=3, b=1 (x1 has 3 different radial cells, x2 has 1 different radial cell)
                        if (board[x1][1] == color and board[x1][2] == color and board[x1][3] == color and
                                board[x2][1] == color):
                            return color
                        # Case 3: a=4, b=0 (x1 has 4 different radial cells, x2 has no different radial cell)
                        if (board[x1][1] == color and board[x1][2] == color and board[x1][3] == color and
                                board[x1][4] == color):
                            return color
        if np.all(board != 0):
            return 2  # Draw
        return 0

    pg.init()
    surface = gp2.draw_board()

    board = np.zeros((16, 10), dtype=int)
    running = True
    gameover = False

    while running:

        for event in pg.event.get():

            if event.type == pg.QUIT:
                running = False

        if not gameover:
            [row, col] = ai_move1(board, 1)  # First group is assigned to be black
            print("black", row, col)
            if board[row, col] == 0:
                board[row, col] = 1
                gp2.draw_stone(surface, [row, col], 1)
            gameover = check_winner(board)

        if not gameover:
            [row, col] = ai_move2(board, -1)  # Second group is assigned to be white
            print("white", row, col)
            if board[row, col] == 0:
                board[row, col] = -1
                gp2.draw_stone(surface, [row, col], -1)
            gameover = check_winner(board)

        if gameover:
            print_winner(surface, gameover)

    pg.quit()


if __name__ == '__main__':
    battle(gp1.ai_move, gp2.ai_move)
