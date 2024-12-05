import numpy as np
import pygame as pg
import random
import time

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

# Convert clicked position to board coordinates (th, r)
def click2index(pos):
    dist = np.sqrt((pos[0] - CENTER) ** 2 + (pos[1] - CENTER) ** 2)
    if dist < W_SIZE / 2 - PAD + 0.25 * SEP_R:
        th, r = round(np.arctan2((pos[1] - CENTER), (pos[0] - CENTER)) / SEP_TH), round(dist / SEP_R)

        if r == 0:
            if not all(FIRST_COLUMN_FILLED):
                if not FIRST_COLUMN_FILLED[th]:
                    return (th, r)
            return False
        else:
            return (th, r)
    return False

# Draw the stone on the board
def draw_stone(surface, pos, color=0):
    x = CENTER + pos[1] * SEP_R * np.cos(pos[0] * SEP_TH)
    y = CENTER + pos[1] * SEP_R * np.sin(pos[0] * SEP_TH)
    if color == 1:
        pg.draw.circle(surface, COLOR_BLACK, (x, y), PIECE_RADIUS * (1 + 2 * pos[1] / RADIAL_SPAN), 0)
    elif color == -1:
        pg.draw.circle(surface, COLOR_WHITE, (x, y), PIECE_RADIUS * (1 + 2 * pos[1] / RADIAL_SPAN), 0)
    pg.display.update()

# Display the game result
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

def check_sequence(board, start_th, start_r, delta_th, delta_r, color):
    first=False
    if delta_th==1 and delta_r==0 and start_r==0:
        for i in range(ANGULAR_SPAN):
            board[i][0]=color
        first=True
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
                        FIRST_COLUMN_FILLED[x1] = True
                        FIRST_COLUMN_FILLED[x2] = True
                        return color
                    # Case 2: a=3, b=1 (x1 has 3 different radial cells, x2 has 1 different radial cell)
                    if (board[x1][1] == color and board[x1][2] == color and board[x1][3] == color and
                            board[x2][1] == color):
                        FIRST_COLUMN_FILLED[x1] = True
                        FIRST_COLUMN_FILLED[x2] = True
                        return color
                    # Case 3: a=4, b=0 (x1 has 4 different radial cells, x2 has no different radial cell)
                    if (board[x1][1] == color and board[x1][2] == color and board[x1][3] == color and
                            board[x1][4] == color):
                        FIRST_COLUMN_FILLED[x1] = True
                        return color
    if np.all(board != 0):
        return 2  # Draw
    return 0

# Function to set all first column positions to a player's color
def set_first_column(board, color):
    for th in range(ANGULAR_SPAN):
        board[th][0] = color
        FIRST_COLUMN_FILLED[th] = True

# Computer's move (random strategy)
def ai_move(board, color):
    possible_moves = []
    for r in range(RADIAL_SPAN):
        for th in range(ANGULAR_SPAN):
            if board[th][r] == 0:
                if r == 0:
                    if not all(FIRST_COLUMN_FILLED):
                        possible_moves.append((th, r))
                else:
                    possible_moves.append((th, r))
    if possible_moves:
        move = random.choice(possible_moves)
        return move
    return None

# Main game loop
def main(player_is_black=True):
    surface = draw_board()  # Initialize the board
    board = np.zeros((ANGULAR_SPAN, RADIAL_SPAN), dtype=int)  # Initialize the board matrix
    running = True
    gameover = False
    player = 1 if player_is_black else -1  # Player's color (1 for black, -1 for white)

    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEBUTTONDOWN and not gameover:
                indx = click2index(event.pos)
                print(indx)
                if indx and board[indx[0]][indx[1]] == 0:
                    if indx[1] == 0:
                        set_first_column(board, player)
                        for th in range(ANGULAR_SPAN):
                            draw_stone(surface, (th, 0), color=player)
                    else:
                        board[indx[0]][indx[1]] = player
                        draw_stone(surface, indx, color=player)
                    winner = check_winner(board)
                    if winner != 0:
                        print_winner(surface, winner)
                        gameover = True
                    player = -player

        if not gameover and player == -1:
            move = computer_move(board, player)
            if move:
                if move[1] == 0:
                    set_first_column(board, player)
                    for th in range(ANGULAR_SPAN):
                        draw_stone(surface, (th, 0), color=player)
                else:
                    board[move[0]][move[1]] = player
                    draw_stone(surface, move, color=player)
                winner = check_winner(board)
                if winner != 0:
                    print_winner(surface, winner)
                    gameover = True
                player = 1

        pg.display.update()

    pg.quit()

# Run the game
if __name__ == '__main__':
    main(player_is_black=True)  # Default is black for the player