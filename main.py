import math
from collections import defaultdict

import pygame
from random import choices, randint
import itertools

# --- Paramètres de base ---
GRID_SIZE = 6  # 6x6
CELL_SIZE = 100  # taille d'une case en pixels
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE
WIN_LEARNING_COUNT = 0

# Couleurs
WHITE = (255, 255, 255)
GRAY = (220, 220, 220)
BLUE = (0, 150, 255)
BLACK = (0, 0, 0)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("pingouinRL")
clock = pygame.time.Clock()

# --- Chargement des images ---
penguin_img = pygame.image.load("assets/penguin.png")
water_img = pygame.image.load("assets/water.png")
igloo_img = pygame.image.load("assets/igloo.png")

# redimensionner
penguin_img = pygame.transform.scale(penguin_img, (CELL_SIZE, CELL_SIZE))
water_img = pygame.transform.scale(water_img, (CELL_SIZE, CELL_SIZE))
igloo_img = pygame.transform.scale(igloo_img, (CELL_SIZE, CELL_SIZE))


# Gestion de la q-table
class q_table:
    # Init la q-table avec des valeurs random
    def __init__(self, states, actions):
        # Dico de la forme {((x,y),action):q_value}
        self.table = {}
        self.actions = actions

        for state in states:
            for action in actions:
                self.table[(tuple(state), action)] = 0.0

    # Renvoie le prochain état selon la politique epsilon greedy
    def next_move(self, state, actions, epsilon):
        x = choices([0, 1], weights=[1 - epsilon, epsilon])[0]

        # Prendre le plus grand Q
        if x == 0:
            best = self.table[(tuple(state), actions[0])]
            best_action = actions[0]

            for action in actions:
                temp = self.table[(tuple(state), action)]
                if temp > best:
                    best = temp
                    best_action = action
            return best_action
        else:
            x = randint(0, len(actions) - 1)
            return actions[x]


# --- Création de la grille ---
class Environment:
    def __init__(self, size):
        self.size = size
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]
        # positions des flaques (éviter le coin de départ et celui de l’igloo)
        self.waters = []
        for _ in range(randint(5, 8)):
            x, y = randint(0, self.size - 1), randint(0, self.size - 1)
            if [x, y] not in ([0, 0], [self.size - 1, self.size - 1]):
                self.waters.append([x, y])
        self.goal = [self.size - 1, self.size - 1]

    def is_water(self):
        if [self.agent_pos[0], self.agent_pos[1]] in self.waters:
            self.agent_pos = [0, 0]
            return True
        return False

    def win(self):
        if [self.agent_pos[0], self.agent_pos[1]] == self.goal:
            self.agent_pos = [0, 0]
            return True
        return False

    def move(self, action):
        if action == "UP":
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == "DOWN":
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == "LEFT":
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == "RIGHT":
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        if self.is_water():
            return "water"

        if self.win():
            return "win"

    def draw(self, surface):
        surface.fill(WHITE)
        # grille
        for x in range(0, WIDTH, CELL_SIZE):
            pygame.draw.line(surface, GRAY, (x, 0), (x, HEIGHT))
        for y in range(0, HEIGHT, CELL_SIZE):
            pygame.draw.line(surface, GRAY, (0, y), (WIDTH, y))
        # flaques
        for wx, wy in self.waters:
            surface.blit(water_img, (wx * CELL_SIZE, wy * CELL_SIZE))
        # igloo
        gx, gy = self.goal
        surface.blit(igloo_img, (gx * CELL_SIZE, gy * CELL_SIZE))
        # pingouin
        ax, ay = self.agent_pos
        surface.blit(penguin_img, (ax * CELL_SIZE, ay * CELL_SIZE))

    def get_possible_moves(self, pos):
        moves = ["LEFT", "RIGHT", "UP", "DOWN"]

        if pos[0] == 0:
            moves.remove("LEFT")

        if pos[0] == 5:
            moves.remove("RIGHT")

        if pos[1] == 0:
            moves.remove("UP")

        if pos[1] == 5:
            moves.remove("DOWN")

        return moves


# plays a game to learn
def learn(qtable, env, alpha, gamma, epsilon):
    steps = 0
    res = ""

    while res != "win" and steps <= 100:
        possible_moves = env.get_possible_moves(env.agent_pos)

        # Garde l'état avant de jouer en mémoire => erreur sans le copy car dcp ca prend la ref et ca reste l'état courant alors qu'on veut l'état avant de jouer
        s = env.agent_pos.copy()

        # Calcule et joue le prochain coup
        next_move = qtable.next_move(env.agent_pos, possible_moves, epsilon)
        res = env.move(next_move)

        steps += 1

        # Choppe le reward
        if res == "win":
            reward = 1
        elif res == "water":
            reward = -1
        else:
            reward = 0

        reward += - steps * 0.01

        # Donne le meilleur q_val possible à partir de s'
        current_q = qtable.table[(tuple(s), next_move)]
        bestfuture_qval = max([qtable.table[(tuple(env.agent_pos), a_prime)] for a_prime in env.get_possible_moves(env.agent_pos)])

        qtable.table[(tuple(s), next_move)] = (1 - alpha) * current_q + alpha * (reward + gamma * bestfuture_qval)

    return steps, res


# --- Boucle principale ---
env = Environment(GRID_SIZE)
running = True

s = [x for x in range(env.size)]
qtable = q_table(list(itertools.product(s, s)), ["UP", "DOWN", "LEFT", "RIGHT"])
alpha = 0.9
gamma = 0.95
epsilon = 1.0

# Plays games to learn
for i in range(0, 5000):
    r = learn(qtable, env, alpha, gamma, epsilon)[1]
    if (r == "win"):
        WIN_LEARNING_COUNT += 1

    epsilon = max(0.1, 1 - 0.001 * i)
    alpha = max(0.05, alpha * 0.999)

print("#won during learning is ")
print(WIN_LEARNING_COUNT)

env.agent_pos = [0, 0]
end = False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Détermine la prochaine action
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE and not end:
                possible_moves = env.get_possible_moves(env.agent_pos)

                # Garde l'état avant de jouer en mémoire
                s = env.agent_pos

                # Calcule et joue le prochain coup
                next_move = qtable.next_move(env.agent_pos, possible_moves, epsilon)
                res = env.move(next_move)

                #Petit code pas très propre pour voir le pingouin aller sur l'igloo au lieu de reset direct sans tout changer
                if res == "win":
                    print("HE WON")
                    end = True
                    env.agent_pos = env.goal.copy()
            else:
                end = False
                env.agent_pos = [0,0]

    env.draw(screen)
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
