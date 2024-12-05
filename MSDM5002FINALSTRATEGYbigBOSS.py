####################################################################################################################
####################################################################################################################
####################################################################################################################
##########   MSDM5002FINALSTRATEGYbigBOSS.py用于检测胜利的check_winner函数放在了MCTS类中，在代码的177-235行   ##############
##########   MSDM5002FINALSTRATEGYbigBOSS.py用于检测胜利的check_winner函数放在了MCTS类中，在代码的177-235行   ##############
##########   MSDM5002FINALSTRATEGYbigBOSS.py用于检测胜利的check_winner函数放在了MCTS类中，在代码的177-235行   ##############
####################################################################################################################
####################################################################################################################
####################################################################################################################

import numpy as np
import pygame as pg
import random
import time
import copy
import math

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
#Sync with first column
def set_first_column(board, color):
    for th in range(ANGULAR_SPAN):
        FIRST_COLUMN_FILLED[th] = True


####################################################################################################################
class Board(object):
    """
    board for game
    """
    def __init__(self, width=10, height=16, n_in_row=5):
        self.width = width
        self.height = height
        self.states = {}
        self.last_change = {"last": -1}
        self.last_last_change = {"last_last": -1}
        self.n_in_row = n_in_row
        self.steps = 0
    def init_board(self):
        self.availables = list(range(self.width * self.height))
        for m in self.availables:
            self.states[m] = 0  # 0表示空
    def move_to_location(self, move):
        h = move // self.width
        w = move % self.width
        return [h, w]
    def location_to_move(self, location):
        if (len(location) != 2):
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if (move not in range(self.width * self.height)):
            return -1
        return move
    def update(self, player, move):  # 更新棋盘
        self.states[move] = player
        self.availables.remove(move)
        self.last_last_change["last_last"] = self.last_change["last"]
        self.last_change["last"] = move
        self.steps += 1



        ####################################################################################################################
class MCTS(object):
    """
    AI player, use Monte Carlo Tree Search with UCB
    """
    def __init__(self, board, play_turn, n_in_row=5, time=5, max_actions=3000):
        self.board = board
        self.play_turn = play_turn  # 出手顺序
        self.calculation_time = float(time)  # 最大运算时间
        self.max_actions = max_actions  # 每次模拟最步数
        self.n_in_row = n_in_row
        self.player = play_turn[0]  # 第一个总是电脑
        self.confident = 2.33  # UCB常数 1.96
        self.plays = {}  # 记录模拟次数，键形如(player, move)
        self.wins = {}  # 记录获胜次数
        self.max_depth = 1
        self.skip = False

    def check_sequence(board, start_th, start_r, delta_th, delta_r, color, length=5):
        if delta_th == 1 and delta_r == 0 and start_r == 0:
            # Special handling for circular direction at center
            return False
        if delta_th == 1 and delta_r != 0:
            for i in range(length):
                th = (start_th + delta_th * i) % ANGULAR_SPAN
                r = start_r + delta_r * i
                if r <= 0 or r >= RADIAL_SPAN or board[th][r] != color:
                    return False
        else:
            for i in range(length):
                th = (start_th + delta_th * i) % ANGULAR_SPAN
                r = start_r + delta_r * i
                if r < 0 or r >= RADIAL_SPAN or board[th][r] != color:
                    return False
        return True
    def check_winner(self,board):
        board1 = np.array([board.states[key] for key in range(160)]).reshape(16, 10)
        directions = [
            (0, 1),  # Radial
            (1, 0),  # Circular
            (1, 1),  # Diagonal increasing theta and r
            (1, -1)  # Diagonal increasing theta, decreasing r
        ]
        for th in range(ANGULAR_SPAN):
            for r in range(RADIAL_SPAN):
                if board1[th][r] == 0:
                    continue
                color = board1[th][r]
                for delta_th, delta_r in directions:
                    if MCTS.check_sequence(board1, th, r, delta_th, delta_r, color):
                        return color  # Return the winner's color
        # Special win condition involving the center point (0, 0)
        for x2 in range(ANGULAR_SPAN):
            x1 = (x2 + 8) % ANGULAR_SPAN
            for color in [1, -1]:
                # Check if the center point and adjacent cells form a valid combination
                if board1[x2][0] == color:
                    # Case 1: a=2, b=2
                    if (board1[x1][1] == color and board1[x1][2] == color and
                            board1[x2][1] == color and board1[x2][2] == color):
                        FIRST_COLUMN_FILLED[x1] = True
                        FIRST_COLUMN_FILLED[x2] = True
                        return color
                    # Case 2: a=3, b=1
                    if (board1[x1][1] == color and board1[x1][2] == color and board1[x1][3] == color and
                            board1[x2][1] == color):
                        FIRST_COLUMN_FILLED[x1] = True
                        FIRST_COLUMN_FILLED[x2] = True
                        return color
                    # Case 3: a=4, b=0
                    if (board1[x1][1] == color and board1[x1][2] == color and board1[x1][3] == color and
                            board1[x1][4] == color):
                        FIRST_COLUMN_FILLED[x1] = True
                        return color
        if np.all(board1 != 0):
            return 2  # Draw
        return 0  # No winner, game continues
    def get_player(self, players):
        p = players.pop(0)
        players.append(p)
        return p
    def get_action(self):  # return move
        if len(self.board.availables) == 1:
            return self.board.availables[0]
        # 重置plays和wins
        self.plays = {}
        self.wins = {}
        self.skip = False
        simulations = 0
        begin = time.time()
        while time.time() - begin < self.calculation_time:
            board_copy = copy.deepcopy(self.board)
            play_turn_copy = copy.deepcopy(self.play_turn)
            self.run_simulation(board_copy, play_turn_copy)
            simulations += 1
        # print("total simulations=", simulations)
        self.skip = self.skipf(self.board)
        move = self.select_one_move(self.board)  # 落子
        location = self.board.move_to_location(move)
        print('Maximum depth searched:', self.max_depth)
        print("6.0AI move: %d,%d\n" % (location[0], location[1]))
        return move
    def run_simulation(self, board, play_turn):
        """
        MCTS main process
        """
        plays = self.plays
        wins = self.wins
        availables = board.availables
        player = self.get_player(play_turn)  # 获取当前出手的玩家
        visited_states = set()  # 记录当前路径上的全部着法
        winner = 0
        expand = True
        # Simulation
        for t in range(1, self.max_actions + 1):
            # Selection
            # 如果所有着法都有统计信息，则获取UCB最大的着法
            if all(plays.get((player, move)) for move in availables):
                log_total = math.log(
                    sum(plays[(player, move)] for move in availables))
                value, move = max(
                    ((wins[(player, move)] / plays[(player, move)]) +
                     math.sqrt(self.confident * log_total / plays[(player, move)]), move)
                    for move in availables)
            else:
                adjacents = []
                if len(availables) > self.n_in_row:
                    adjacents = self.adjacent_moves(board, player, plays)  # 没有统计信息的邻近位置

                if len(adjacents):
                    move = random.choice(adjacents)
                else:
                    peripherals = []
                    for move in availables:
                        if not plays.get((player, move)):
                            peripherals.append(move)  # 没有统计信息的其他位置
                    move = random.choice(peripherals)
            board.update(player, move)
            # Expand
            # 每次模拟最多扩展一次，每次扩展只增加一个着法
            if expand and (player, move) not in plays:
                expand = False
                plays[(player, move)] = 0
                wins[(player, move)] = 0
                if t > self.max_depth:
                    self.max_depth = t
            visited_states.add((player, move))
            is_full = not len(availables)
            win = self.check_winner(board) != 0
            winner = 0
            if win:
                winner = self.check_winner(board)
            if is_full or win:  # 没有落子位置或有玩家获胜
                break
            player = self.get_player(play_turn)
        # Back-propagation
        for player, move in visited_states:
            if (player, move) not in plays:
                continue
            plays[(player, move)] += 1  # 当前路径上所有着法的模拟次数加1
            if player == winner:
                wins[(player, move)] += 1  # 获胜玩家的所有着法的胜利次数加1
    def skipf(self, board):
        print("skipf")
        indic = self.checkai4(board)  # ai4
        if len(indic) != 0:  # ai有4
            print("ai下4")
            return list(indic)  # 直接下
        else:  # ai没有4
            indic2 = self.checkp4(board)  # 玩家4
            if len(indic2) != 0:  # 玩家有4
                print("堵玩家4")
                return list(indic2)  # 堵玩家
            else:  # 玩家没有4
                indic3 = self.checkp3(board)  # 玩家活3
                indic4 = self.checkai3(board)  # ai活3
                if len(indic4) != 0:  # ai有活3
                    print("ai有活3")
                    return list(indic4)  # 直接下
                elif len(indic3) != 0:  # ai没有活3，玩家有活3
                    print("ai没有活3，玩家有活3")
                    return list(indic3)  # 堵玩家
                else:  # ai没有活3，玩家没有活3
                    # 仅考虑玩家或ai一次不产生多个禁手点
                    fbp = self.checkpforbidp(board)  # 玩家禁手
                    fbai = self.checkpforbidai(board)  # ai禁手
                    if len(fbp[0]) != 0 and len(fbai[0]) == 0:  # 玩家有禁手，ai无禁手
                        print("堵玩家禁手")
                        return list(fbp[0])  # 堵玩家
                    elif len(fbp[0]) == 0 and len(fbai[0]) != 0:  # 玩家无禁手，ai有禁手
                        print("ai走禁手")
                        return list(fbai[0])  # 走禁手
                    elif len(fbp[0]) != 0 and len(fbai[0]) != 0:  # 均有禁手
                        if 'strong' == fbp[1][0] and 'weak' == fbai[1][0]:  # 玩家强禁手，ai弱禁手
                            print("玩家强禁手，ai弱禁手,堵玩家")
                            return list(fbp[0])  # 堵玩家
                        else:  # 其余情况
                            print("其余情况,走ai禁手")
                            return list(fbai[0])  # 走ai禁手
                    else:  # 均无禁手
                        pt = self.check_check_fbai(board)  # ai潜力
                        if len(pt) != 0:  # ai有潜力点
                            print("ai走潜力点")
                            return list(pt)  # 走潜力点
                        else:
                            return False
    def select_one_move(self, board):
        if self.skip:
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in self.skip)
            for move in self.skip:
                print("move:",move)
        elif board.steps > 9:
            print("没有好策略，最大可能选择周围两圈位置")
            limited = self.adjacent2(board) + self.adjacent3(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited)
        else:
            print("没有好策略，选择周围一圈可能位置")
            limited = self.adjacent2(board)
            percent_wins, move = max(
                (self.wins.get((self.player, move), 0) /
                 self.plays.get((self.player, move), 1),
                 move)
                for move in limited)
        return move
    def adjacent_moves(self, board, player, plays):
        """
        获取当前棋局中所有棋子的邻近位置中没有统计信息的位置
        """
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height
        for m in moved:
            h = m // width
            w = m % width
            if w == width - 1:
                adjacents.add(m - 1)  # 右边界
                adjacents.add((m + width)%160)
                adjacents.add((m - width)%160)
                adjacents.add((m - 1 + width)%160)
                adjacents.add((m - 1 - width)%160)
            elif w == 0:
                adjacents.add(m + 1)  # 左边界
                adjacents.add((m + 1 + width)%160)
                adjacents.add((m + 1 - width)%160)
                adjacents.add((m + width)%160)
                adjacents.add((m - width)%160)
            else:
                adjacents.add(m + 1)
                adjacents.add(m - 1)
                adjacents.add((m + 1 + width) % 160)
                adjacents.add((m - 1 + width) % 160)
                adjacents.add((m + 1 - width) % 160)
                adjacents.add((m - 1 - width) % 160)
                adjacents.add((m + width) % 160)
                adjacents.add((m - width) % 160)

        adjacents = list(set(adjacents) - set(moved))
        for move in adjacents:
            if plays.get((player, move)):
                adjacents.remove(move)
        return adjacents
    def checkp4(self, board):
        """
        检查玩家4
        """
        moved = set(range(board.width * board.height)) - set(board.availables)
        array_2d = np.array([board.states[key] for key in range(160)]).reshape(16, 10)  # 值矩阵 -1,0,1
        n = board.height
        tent = board.last_change["last"]
        if tent == -1:
            return []
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]
        player = array_2d[i][j]
        target410 = [0, player, player, player, player]
        target411 = [player, player, player, player, 0]
        target42 = [player, player, 0, player, player]
        target430 = [player, 0, player, player, player]
        target431 = [player, player, player, 0, player]
        window_size = len(target410)
        results = set()

        # 定义辅助函数
        def add_if_valid(row, col):
            if 0<=row<16 and 0<=col<10:
                results.add(board19[row][col])

        # 水平方向检查
        indexlist1=[]
        if 0 <= i < 16:
            for k in range(-4, 5):
                if j + k < 0:
                    indexlist1.append([(i+8)%16,-(j+k)])
                elif 0 <= j + k:
                    indexlist1.append([i, j + k])
        A1 = [array_2d[m[0], m[1]] if 0<=m[1]<10 else 0 for m in indexlist1]
        if len(A1) >= window_size:
            for a in range(len(A1) - window_size + 1):
                if A1[a:a + window_size] == target410:
                    add_if_valid(i, j - (4 - a))
                if A1[a:a + window_size] == target411:
                    add_if_valid(i, j + a)
                if A1[a:a + window_size] == target42:
                    add_if_valid(i, j - (2 - a))
                if A1[a:a + window_size] == target430:
                    add_if_valid(i, j - (3 - a))
                if A1[a:a + window_size] == target431:
                    add_if_valid(i, j - (1 - a))

        # 垂直方向检查
        indexlist2 = [[(i + k)%16, j] for k in range(-4, 5) if 0<=j<10]
        A2 = [array_2d[m[0], m[1]] for m in indexlist2]
        if len(A2) >= window_size:
            for a in range(len(A2) - window_size + 1):
                if A2[a:a + window_size] == target410:
                    add_if_valid((i - (4 - a))%16, j)
                if A2[a:a + window_size] == target411:
                    add_if_valid((i + a)%16, j)
                if A2[a:a + window_size] == target42:
                    add_if_valid((i - (2 - a)%16), j)
                if A2[a:a + window_size] == target430:
                    add_if_valid((i - (3 - a))%16, j)
                if A2[a:a + window_size] == target431:
                    add_if_valid((i - (1 - a))%16, j)

        # 主对角线方向检查
        indexlist3 = [[(i + k)%16, j + k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist3:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A3 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist3]
        if len(A3) >= window_size:
            for a in range(len(A3) - window_size + 1):
                if A3[a:a + window_size] == target410:
                    add_if_valid((i - (4 - a)) % 16, j - (4 - a))
                if A3[a:a + window_size] == target411:
                    add_if_valid((i + a) % 16, j + a)
                if A3[a:a + window_size] == target42:
                    add_if_valid((i - (2 - a))%16, j - (2 - a))
                if A3[a:a + window_size] == target430:
                    add_if_valid((i - (3 - a))%16, j - (3 - a))
                if A3[a:a + window_size] == target431:
                    add_if_valid((i - (1 - a))%16, j - (1 - a))
        array_2d[x][y] = player

        # 副对角线方向检查
        indexlist4 = [[(i + k)%16, j - k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist4:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A4 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist4]
        if len(A4) >= window_size:
            for a in range(len(A4) - window_size + 1):
                if A4[a:a + window_size] == target410:
                    add_if_valid((i - (4 - a)) % 16, j + (4 - a))
                if A4[a:a + window_size] == target411:
                    add_if_valid((i + a) % 16, j - a)
                if A4[a:a + window_size] == target42:
                    add_if_valid((i - (2 - a))%16, j + (2 - a))
                if A4[a:a + window_size] == target430:
                    add_if_valid((i - (3 - a))%16, j + (3 - a))
                if A4[a:a + window_size] == target431:
                    add_if_valid((i - (1 - a))%16, j + (1 - a))
        array_2d[x][y] = player

        results=results-moved
        return results
    def checkai4(self, board):
        """
        检查ai4
        """
        moved = set(range(board.width * board.height)) - set(board.availables)
        array_2d = np.array([board.states[key] for key in range(160)]).reshape(16, 10)  # 值矩阵 -1,0,1
        board1 = array_2d
        tent = board.last_last_change["last_last"]
        if tent == -1:
            return []
        i, j = board.move_to_location(tent)
        player = board1[i][j]
        target410 = [0, player, player, player, player]
        target411 = [player, player, player, player, 0]
        target42 = [player, player, 0, player, player]
        target430 = [player, 0, player, player, player]
        target431 = [player, player, player, 0, player]
        window_size = len(target410)
        results = set()

        # 定义辅助函数
        def add_if_valid(row, col):
            if 0<=row<16 and 0<=col<10:
                results.add(board19[row][col])

        # 水平方向检查
        indexlist1 = []
        if 0 <= i < 16:
            for k in range(-4, 5):
                if j + k < 0:
                    indexlist1.append([(i + 8) % 16, -(j + k)])
                elif 0 <= j + k:
                    indexlist1.append([i, j + k])

        A1 = [board1[m[0], m[1]] if 0<=m[1]<10 else 0 for m in indexlist1]
        if len(A1) >= window_size:
            for a in range(len(A1) - window_size + 1):
                if A1[a:a + window_size] == target410:
                    add_if_valid(i, j - (4 - a))
                if A1[a:a + window_size] == target411:
                    add_if_valid(i, j + a)
                if A1[a:a + window_size] == target42:
                    add_if_valid(i, j - (2 - a))
                if A1[a:a + window_size] == target430:
                    add_if_valid(i, j - (3 - a))
                if A1[a:a + window_size] == target431:
                    add_if_valid(i, j - (1 - a))

        # 垂直方向检查
        indexlist2 = [[(i + k)%16, j] for k in range(-4, 5) if 0<=j<10]
        A2 = [array_2d[m[0], m[1]] for m in indexlist2]
        if len(A2) >= window_size:
            for a in range(len(A2) - window_size + 1):
                if A2[a:a + window_size] == target410:
                    add_if_valid((i - (4 - a)) % 16, j)
                if A2[a:a + window_size] == target411:
                    add_if_valid((i + a) % 16, j)
                if A2[a:a + window_size] == target42:
                    add_if_valid((i - (2 - a) % 16), j)
                if A2[a:a + window_size] == target430:
                    add_if_valid((i - (3 - a)) % 16, j)
                if A2[a:a + window_size] == target431:
                    add_if_valid((i - (1 - a)) % 16, j)

        # 主对角线方向检查
        indexlist3 = [[(i + k) % 16, j + k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist3:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A3 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist3]
        if len(A3) >= window_size:
            for a in range(len(A3) - window_size + 1):
                if A3[a:a + window_size] == target410:
                    add_if_valid((i - (4 - a)) % 16, j - (4 - a))
                    add_if_valid((i - (4 - a) + 1) % 16, j - (4 - a) + 1)
                    add_if_valid((i - (4 - a) - 1) % 16, j - (4 - a) - 1)
                if A3[a:a + window_size] == target411:
                    add_if_valid((i + a + 1) % 16, j + a + 1)
                    add_if_valid((i + a) % 16, j + a)
                    add_if_valid((i + a - 1) % 16, j + a - 1)
                if A3[a:a + window_size] == target42:
                    add_if_valid((i - (2 - a)) % 16, j - (2 - a))
                if A3[a:a + window_size] == target430:
                    add_if_valid((i - (3 - a)) % 16, j - (3 - a))
                if A3[a:a + window_size] == target431:
                    add_if_valid((i - (1 - a)) % 16, j - (1 - a))
        array_2d[x][y] == player

        # 副对角线方向检查
        indexlist4 = [[(i + k) % 16, j - k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist4:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A4 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist4]
        if len(A4) >= window_size:
            for a in range(len(A4) - window_size + 1):
                if A4[a:a + window_size] == target410:
                    add_if_valid((i - (4 - a)) % 16, j + (4 - a))
                    add_if_valid((i - (4 - a) + 1) % 16, j + (4 - a) - 1)
                    add_if_valid((i - (4 - a) - 1) % 16, j + (4 - a) + 1)
                if A4[a:a + window_size] == target411:
                    add_if_valid((i + a) % 16, j - a)
                    add_if_valid((i + a + 1) % 16, j - a - 1)
                    add_if_valid((i + a - 1) % 16, j - a + 1)
                if A4[a:a + window_size] == target42:
                    add_if_valid((i - (2 - a)) % 16, j + (2 - a))
                if A4[a:a + window_size] == target430:
                    add_if_valid((i - (3 - a)) % 16, j + (3 - a))
                if A4[a:a + window_size] == target431:
                    add_if_valid((i - (1 - a)) % 16, j + (1 - a))
        array_2d[x][y] == player

        results=results-moved
        return results
    def checkp3(self, board):
        """
        检查玩家3
        """
        moved = set(range(board.width * board.height)) - set(board.availables)
        array_2d = np.array([board.states[key] for key in range(160)]).reshape(16, 10)  # 值矩阵 -1,0,1
        board1 = array_2d
        n = board.height
        tent = board.last_change["last"]
        if tent == -1:
            return []
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]
        player = array_2d[i][j]
        target30 = [0, player, player, player, 0]
        target31 = [0, player, 0, player, player, 0]
        target32 = [0, player, player, 0, player, 0]
        window_size1 = len(target30)
        window_size2 = len(target31)
        results = set()

        # 定义辅助函数
        def add_if_valid(row, col):
            if 0<=row<16 and 0<=col<10:
                results.add(board19[row][col])

        # 水平方向检查
        indexlist1 = []
        if 0 <= i < 16:
            for k in range(-4, 5):
                if j + k < 0:
                    indexlist1.append([(i+8)%16, -(j + k)])
                elif 0<=j+k:
                    indexlist1.append([i, j + k])
        A1 = [board1[m[0], m[1]] if 0<=m[1]<10 else 0 for m in indexlist1]
        if len(A1) >= window_size1:
            for a in range(len(A1) - window_size1 + 1):
                if A1[a:a + window_size1] == target30:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j + a)
        if len(A1) >= window_size2:
            for a in range(len(A1) - window_size2 + 1):
                if A1[a:a + window_size2] == target31:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j - (2 - a))
                    add_if_valid(i, j + (1 + a))
                if A1[a:a + window_size2] == target32:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j - (1 - a))
                    add_if_valid(i, j + (1 + a))

        # 垂直方向检查
        indexlist2 = [[(i + k)%16, j] for k in range(-4, 5) if 0<=j<10]
        A2 = [array_2d[m[0], m[1]] for m in indexlist2]
        if len(A2) >= window_size1:
            for a in range(len(A2) - window_size1 + 1):
                if A2[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i + a)%16, j)
        if len(A2) >= window_size2:
            for a in range(len(A2) - window_size2 + 1):
                if A2[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i - (2 - a))%16, j)
                    add_if_valid((i + (1 + a))%16, j)
                if A2[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i - (1 - a))%16, j)
                    add_if_valid((i + (1 + a))%16, j)

        # 主对角线方向检查
        indexlist3 = [[(i + k) % 16, j + k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist3:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A3 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist3]
        if len(A3) >= window_size1:
            for a in range(len(A3) - window_size1 + 1):
                if A3[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, (j - (4 - a))%10)
                    add_if_valid((i + a)%16, (j + a)%10)
        if len(A3) >= window_size2:
            for a in range(len(A3) - window_size2 + 1):
                if A3[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j - (4 - a))
                    add_if_valid((i - (2 - a))%16, j - (2 - a))
                    add_if_valid((i + (1 + a))%16, j + (1 + a))
                if A3[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j - (4 - a))
                    add_if_valid((i - (1 - a))%16, j - (1 - a))
                    add_if_valid((i + (1 + a))%16, j + (1 + a))
        array_2d[x][y] == player

        # 副对角线方向检查
        indexlist4 = [[(i + k) % 16, j - k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist4:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A4 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist4]
        if len(A4) >= window_size1:
            for a in range(len(A4) - window_size1 + 1):
                if A4[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i + a)%16, j - a)
        if len(A4) >= window_size2:
            for a in range(len(A4) - window_size2 + 1):
                if A4[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i - (2 - a))%16, j + (2 - a))
                    add_if_valid((i + (1 + a))%16, j - (1 + a))
                if A4[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i - (1 - a))%16, j + (1 - a))
                    add_if_valid((i + (1 + a))%16, j - (1 + a))
        array_2d[x][y] == player

        results=results-moved
        return results
    def checkai3(self, board):
        """
        检查AI3
        """
        moved = set(range(board.width * board.height)) - set(board.availables)
        array_2d = np.array([board.states[key] for key in range(160)]).reshape(16, 10)  # 值矩阵 -1,0,1
        board1 = array_2d
        n = board.height
        tent = board.last_last_change["last_last"]
        if tent == -1:
            return []
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]
        player = array_2d[i][j]
        target30 = [0, player, player, player, 0]
        target31 = [0, player, 0, player, player, 0]
        target32 = [0, player, player, 0, player, 0]
        window_size1 = len(target30)
        window_size2 = len(target31)
        results = set()

        # 定义辅助函数
        def add_if_valid(row, col):
            if 0<=row<16 and 0<=col<10:
                results.add(board19[row][col])
        # 水平方向检查
        indexlist1 = []
        if 0 <= i < 16:
            for k in range(-4, 5):
                if j + k < 0:
                    indexlist1.append([(i+8)%16, -(j + k)])
                elif 0 <= j + k:
                    indexlist1.append([i, j + k])
        A1 = [board1[m[0], m[1]] if 0<=m[1]<10 else 0 for m in indexlist1]
        if len(A1) >= window_size1:
            for a in range(len(A1) - window_size1 + 1):
                if A1[a:a + window_size1] == target30:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j + a)
        if len(A1) >= window_size2:
            for a in range(len(A1) - window_size2 + 1):
                if A1[a:a + window_size2] == target31:
                    add_if_valid(i, j - (2 - a))
                if A1[a:a + window_size2] == target32:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j - (1 - a))
                    add_if_valid(i, j + (1 + a))

        # 垂直方向检查
        indexlist2 = [[(i + k)%16, j] for k in range(-4, 5) if 0<=j<10]
        A2 = [array_2d[m[0], m[1]] for m in indexlist2]
        if len(A2) >= window_size1:
            for a in range(len(A2) - window_size1 + 1):
                if A2[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i + a)%16, j)
        if len(A2) >= window_size2:
            for a in range(len(A2) - window_size2 + 1):
                if A2[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i - (2 - a))%16, j)
                    add_if_valid((i + (1 + a))%16, j)
                if A2[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i - (1 - a))%16, j)
                    add_if_valid((i + (1 + a))%16, j)

        # 主对角线方向检查
        indexlist3 = [[(i + k) % 16, j + k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist3:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A3 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist3]
        if len(A3) >= window_size1:
            for a in range(len(A3) - window_size1 + 1):
                if A3[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, (j - (4 - a))%10)
                    add_if_valid((i + a)%16, (j + a)%10)
        if len(A3) >= window_size2:
            for a in range(len(A3) - window_size2 + 1):
                if A3[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j - (4 - a))
                    add_if_valid((i - (2 - a))%16, j - (2 - a))
                    add_if_valid((i + (1 + a))%16, j + (1 + a))
                if A3[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j - (4 - a))
                    add_if_valid((i - (1 - a))%16, j - (1 - a))
                    add_if_valid((i + (1 + a))%16, j + (1 + a))
        array_2d[x][y] == player

        # 副对角线方向检查
        indexlist4 = [[(i + k) % 16, j - k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist4:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A4 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist4]
        if len(A4) >= window_size1:
            for a in range(len(A4) - window_size1 + 1):
                if A4[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i + a)%16, j - a)
        if len(A4) >= window_size2:
            for a in range(len(A4) - window_size2 + 1):
                if A4[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i - (2 - a))%16, j + (2 - a))
                    add_if_valid((i + (1 + a))%16, j - (1 + a))
                if A4[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i - (1 - a))%16, j + (1 - a))
                    add_if_valid((i + (1 + a))%16, j - (1 + a))
        array_2d[x][y] == player

        results=results-moved
        return results
    def adjacent2(self, board):  # 周围一圈
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        adjacents = set()
        width = board.width
        height = board.height
        for m in moved:
            h = m // width
            w = m % width
            if w == width - 1:
                adjacents.add(m - 1)  # 右边界
                adjacents.add((m + width) % 160)
                adjacents.add((m - width) % 160)
                adjacents.add((m - 1 + width) % 160)
                adjacents.add((m - 1 - width) % 160)
            elif w == 0:
                adjacents.add(m + 1)  # 左边界
                adjacents.add((m + 1 + width) % 160)
                adjacents.add((m + 1 - width) % 160)
                adjacents.add((m + width) % 160)
                adjacents.add((m - width) % 160)
            else:
                adjacents.add(m + 1)
                adjacents.add(m - 1)
                adjacents.add((m + 1 + width) % 160)
                adjacents.add((m - 1 + width) % 160)
                adjacents.add((m + 1 - width) % 160)
                adjacents.add((m - 1 - width) % 160)
                adjacents.add((m + width) % 160)
                adjacents.add((m - width) % 160)
        adjacents = list(set(adjacents) - set(moved))
        return adjacents
    def adjacent3(self, board):  # 周围第二圈
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        moved += self.adjacent2(board)
        adjacents = set()
        width = board.width
        height = board.height
        for m in moved:
            h = m // width
            w = m % width
            if w == width - 1:
                adjacents.add(m - 1)  # 右边界
                adjacents.add((m + width)%160)
                adjacents.add((m - width)%160)
                adjacents.add((m - 1 + width)%160)
                adjacents.add((m - 1 - width)%160)
            elif w == 0:
                adjacents.add(m + 1)  # 左边界
                adjacents.add((m + 1 + width)%160)
                adjacents.add((m + 1 - width)%160)
                adjacents.add((m + width)%160)
                adjacents.add((m - width)%160)
            else:
                adjacents.add(m + 1)
                adjacents.add(m - 1)
                adjacents.add((m + 1 + width) % 160)
                adjacents.add((m - 1 + width) % 160)
                adjacents.add((m + 1 - width) % 160)
                adjacents.add((m - 1 - width) % 160)
                adjacents.add((m + width) % 160)
                adjacents.add((m - width) % 160)

        adjacents = list(set(adjacents) - set(moved))
        return adjacents
    def checkpforbidp(self, board):
        """
        检查玩家禁手
        """

        pool = self.adjacent2(board) + self.adjacent3(board)
        forbidmove = set()
        attribute = list()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(-self.player, i)
            g3 = self.checkp3(board_copy)
            g4 = self.checkp4(board_copy)
            if len(g3) > 3:
                forbidmove.add(i)
                attribute.append('weak')
            elif len(g4) >= 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)
                attribute.append('strong')
        return forbidmove, attribute
    def checkpforbidai(self, board):
        """
        检查AI禁手
        """

        pool = self.adjacent2(board) + self.adjacent3(board)
        forbidmove = set()
        attribute = list()
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(self.player, i)
            board_copy.last_last_change["last_last"] = board_copy.last_change["last"]
            g3 = self.checkai3_all(board_copy)
            g4 = self.checkai4(board_copy)
            if len(g3) > 3:
                forbidmove.add(i)
                attribute.append('weak')
                # print('find forbid for ai!!')
                # print('forbid:',self.player, 'g3:',g3, 'g4:',g4)
            elif len(g4) >= 2 or (len(g4) > 0 and len(g3) > 0):
                forbidmove.add(i)
                attribute.append('strong')
                # print('find forbid for ai!!')
                # print('forbid:',self.player, 'g3:',g3, 'g4:',g4)
        return forbidmove, attribute
    def checkai3_all(self, board):
        """
        检查AI3，返回所有点
        """

        array_2d = np.array([board.states[key] for key in range(160)]).reshape(16, 10)  # 值矩阵 -1,0,1
        board1 = array_2d
        n = board.height
        tent = board.last_last_change["last_last"]
        if tent == -1:
            return []
        i = board.move_to_location(tent)[0]
        j = board.move_to_location(tent)[1]
        player = array_2d[i][j]
        target30 = [0, player, player, player, 0]
        target31 = [0, player, 0, player, player, 0]
        target32 = [0, player, player, 0, player, 0]
        window_size1 = len(target30)
        window_size2 = len(target31)
        results = set()



        # 定义辅助函数
        def add_if_valid(row, col):
            if 0<=row<16 and 0<=col<10:
                results.add(board19[row][col])

        # 水平方向检查
        indexlist1 = []
        if 0 <= i < 16:
            for k in range(-4, 5):
                if j + k < 0:
                    indexlist1.append([(i+8)%16, -(j + k)])
                elif 0 <= j + k:
                    indexlist1.append([i, j + k])
        A1 = [board1[m[0], m[1]] if 0<=m[1]<10 else 0 for m in indexlist1]
        if len(A1) >= window_size1:
            for a in range(len(A1) - window_size1 + 1):
                if A1[a:a + window_size1] == target30:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j + a)
        if len(A1) >= window_size2:
            for a in range(len(A1) - window_size2 + 1):
                if A1[a:a + window_size2] == target31:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j - (2 - a))
                    add_if_valid(i, j + (1 + a))
                if A1[a:a + window_size2] == target32:
                    add_if_valid(i, j - (4 - a))
                    add_if_valid(i, j - (1 - a))
                    add_if_valid(i, j + (1 + a))

        # 垂直方向检查
        indexlist2 = [[(i + k)%16, j] for k in range(-4, 5) if 0<=j<10]
        A2 = [array_2d[m[0], m[1]] for m in indexlist2]
        if len(A2) >= window_size1:
            for a in range(len(A2) - window_size1 + 1):
                if A2[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i + a)%16, j)
        if len(A2) >= window_size2:
            for a in range(len(A2) - window_size2 + 1):
                if A2[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i - (2 - a))%16, j)
                    add_if_valid((i + (1 + a))%16, j)
                if A2[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j)
                    add_if_valid((i - (1 - a))%16, j)
                    add_if_valid((i + (1 + a))%16, j)

        # 主对角线方向检查
        indexlist3 = [[(i + k) % 16, j + k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist3:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A3 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist3]
        if len(A3) >= window_size1:
            for a in range(len(A3) - window_size1 + 1):
                if A3[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, (j - (4 - a))%10)
                    add_if_valid((i + a)%16, (j + a)%10)
        if len(A3) >= window_size2:
            for a in range(len(A3) - window_size2 + 1):
                if A3[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j - (4 - a))
                    add_if_valid((i - (2 - a))%16, j - (2 - a))
                    add_if_valid((i + (1 + a))%16, j + (1 + a))
                if A3[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j - (4 - a))
                    add_if_valid((i - (1 - a))%16, j - (1 - a))
                    add_if_valid((i + (1 + a))%16, j + (1 + a))
        array_2d[x][y] == player

        # 副对角线方向检查
        indexlist4 = [[(i + k) % 16, j - k] for k in range(-4, 5)]
        x,y=0,0
        for u in indexlist4:
            if u[1]==0 and array_2d[u[0]][u[1]] == player:
                array_2d[u[0]][u[1]]=0
                x,y=u[0],u[1]
                break
        A4 = [array_2d[m[0], m[1]] if 0 <= m[0] < 16 and 0 <= m[1] < 10 else 0 for m in indexlist4]
        if len(A4) >= window_size1:
            for a in range(len(A4) - window_size1 + 1):
                if A4[a:a + window_size1] == target30:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i + a)%16, j - a)
        if len(A4) >= window_size2:
            for a in range(len(A4) - window_size2 + 1):
                if A4[a:a + window_size2] == target31:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i - (2 - a))%16, j + (2 - a))
                    add_if_valid((i + (1 + a))%16, j - (1 + a))
                if A4[a:a + window_size2] == target32:
                    add_if_valid((i - (4 - a))%16, j + (4 - a))
                    add_if_valid((i - (1 - a))%16, j + (1 - a))
                    add_if_valid((i + (1 + a))%16, j - (1 + a))
        array_2d[x][y] == player
        return results
    def check_check_fbai(self, board): #ai潜力
        moved = list(set(range(board.width * board.height)) - set(board.availables))
        pool = self.adjacent2(board) + self.adjacent3(board)
        potential = set()
        for tent in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(-self.player, tent)
            array_2d = np.array([board_copy.states[key] for key in range(160)]).reshape(16, 10)  # 值矩阵 -1,0,1
            board1 = array_2d
            if tent == -1:
                return []
            i = board_copy.move_to_location(tent)[0]
            j = board_copy.move_to_location(tent)[1]
            player = array_2d[i][j]
            target30 = [0, player, player, player, 0]
            window_size1 = len(target30)
            if array_2d[i][j] == player:
                total = 0
                # 水平方向检查
                indexlist1 = []
                if 0 <= i < 16:
                    for k in range(-4, 5):
                        if -10 < j + k < 0:
                            indexlist1.append([(i + 8) % 16, -(j + k)])
                        elif 0 <= j + k < 10:
                            indexlist1.append([i, j + k])
                A1 = [board1[m[0], m[1]] for m in indexlist1]
                if len(A1) >= window_size1:
                    for a in range(len(A1) - window_size1 + 1):
                        if A1[a:a + window_size1] == target30:
                            total += 1

                # 垂直方向检查
                indexlist2 = [[(i + k) % 16, j] for k in range(-4, 5) if 0 <= j < 10]
                A2 = [board1[m[0], m[1]] for m in indexlist2]
                if len(A2) >= window_size1:
                    for a in range(len(A2) - window_size1 + 1):
                        if A2[a:a + window_size1] == target30:
                            total += 1

                # 主对角线方向检查
                indexlist3 = [[(i + k) % 16, j + k] for k in range(-4, 5) if 0 <= j + k < 10]
                x, y = 0, 0
                for u in indexlist3:
                    if u[1] == 0 and array_2d[u[0]][u[1]] == player:
                        array_2d[u[0]][u[1]] = 0
                        x, y = u[0], u[1]
                        break
                A3 = [board1[m[0], m[1]] for m in indexlist3]
                if len(A3) >= window_size1:
                    for a in range(len(A3) - window_size1 + 1):
                        if A3[a:a + window_size1] == target30:
                            total += 1
                board1[x][y] == player
                # 副对角线方向检查
                indexlist4 = [[(i + k) % 16, j - k] for k in range(-4, 5) if 0 <= j - k < 10]
                x, y = 0, 0
                for u in indexlist4:
                    if u[1] == 0 and array_2d[u[0]][u[1]] == player:
                        array_2d[u[0]][u[1]] = 0
                        x, y = u[0], u[1]
                        break
                A4 = [board1[m[0], m[1]] for m in indexlist4]
                if len(A4) >= window_size1:
                    for a in range(len(A4) - window_size1 + 1):
                        if A4[a:a + window_size1] == target30:
                            total += 1
                board1[x][y] == player
            if total>= 2 :
                potential.add(tent)
        if potential:
            return potential
        for i in pool:
            board_copy = copy.deepcopy(board)
            board_copy.update(self.player, i)
            if len(self.checkpforbidai(board_copy)[0]) != 0 :
                potential.add(i)
        potential=potential-set(moved)
        return potential




####################################################################################################################
def ai_move(B, c):
    global  row, col, board19, B_lastlast

    row = None
    col = None
    if np.array_equal(np.zeros((16, 10), dtype=int), B):  # 若棋盘B全空则下中间
        return 0,0

    # 若第一次运行这个函数，创建一个B_lastlast全局变量初始化为全0，用来记录两回合前的状态。先手则第3回合B比B_lastlast多了一白一黑
    # ，579...回合全部多一白一黑；后手则第2回合B比B_lastlast只多了一黑，468...回合全部多一黑一白
    if not hasattr(ai_move, 'is_first_run'):
        B_lastlast = np.zeros((16, 10), dtype=int)
        ai_move.is_first_run = True

    # 每次初始化类
    BOARD = Board()
    BOARD.init_board()
    play_turn = [c, -c]
    AI = MCTS(BOARD, play_turn)

    array_key = np.array(range(160)).reshape(16, 10)  # 键矩阵 0 - 159
    board19 = array_key # 19×19 键棋盘 键矩阵 0 - 159

    # 先更新B上上回合的所有状态
    for i in range(16):
        for j in range(10):
            if B[i, j] != 0 and B_lastlast[i, j] == B[i, j]:
                BOARD.update(B[i, j], array_key[i, j])

    # 再更新B上上回合自己走的，后手第一次落子则不需要更新
    for i in range(16):
        for j in range(10):
            if B_lastlast[i, j] != B[i, j] and B[i, j] == c:
                BOARD.update(B[i, j], array_key[i, j])

    # 再更新B上回合对面走的
    for i in range(16):
        for j in range(10):
            if B_lastlast[i, j] != B[i, j] and B[i, j] == -c:
                BOARD.update(B[i, j], array_key[i, j])

    aimove = AI.get_action()  # 先判断走哪步
    aimove_index = BOARD.move_to_location(aimove)  # 变为坐标
    B_lastlast = B.copy()  # 再更新全局变量B_lastlast，用于两回合后
    return aimove_index




####################################################################################################################
def main_template(player_is_black=True):
    global row, col, board19
    pg.init()
    surface = draw_board()
    board = Board()
    board.init_board()
    running = True
    gameover = False
    if not player_is_black:
        draw_stone(surface, [0, 0], 1)
        board.update(1, 0)
    row = None
    col = None
    colorai = -1 if player_is_black else 1
    play_turn = [colorai, -colorai]
    AI = MCTS(board, play_turn)
    array_key = np.array(range(160)).reshape(16, 10)  # 键矩阵 0 - 128
    board19 = array_key
    while running:
        for event in pg.event.get():  # A for loop to process all the events initialized by the player
            if event.type == pg.QUIT:  # terminate if player closes the game window
                running = False
            if event.type == pg.MOUSEBUTTONDOWN and not gameover:  # detect whether the player is clicking in the window
                (x, y) = event.pos  # check if the clicked position is on the 11x11 center grid
                (th,r)=click2index([x, y])
                if r<=10:
                    row = th
                    col = r
                    move = row * 10 + col
                    if board.states[move] == 0:  # update the board matrix if that position has not been occupied
                        color = 1 if player_is_black else -1
                        board.update(color, move)
                        if move%10==0 and any(FIRST_COLUMN_FILLED):
                            set_first_column(board, color)
                            for i in range(ANGULAR_SPAN):
                                if i !=move//10:
                                    board.update(color, i*10)
                                    board.steps-=1
                        draw_stone(surface, [row, col], color)
                        if AI.check_winner(AI.board) != 0:
                            print_winner(surface, winner=AI.check_winner(AI.board))
                            gameover = True
                        else:
                            color2 = -1 if player_is_black else 1
                            aimove = AI.get_action()
                            board.update(color2, aimove)
                            if aimove % 10 == 0 and any(FIRST_COLUMN_FILLED):
                                set_first_column(board, color2)
                                for i in range(ANGULAR_SPAN):
                                    if i != aimove // 10:
                                        board.update(color2, i * 10)
                                        board.steps-=1
                            draw_stone(surface, board.move_to_location(aimove), color2)
                            if AI.check_winner(AI.board) != 0:
                                print_winner(surface, winner=AI.check_winner(AI.board))
                                gameover = True
    pg.quit()
if __name__ == '__main__':
    main_template(True)