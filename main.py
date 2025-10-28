import math

import pygame
import random
import itertools

# --- Paramètres de base ---
GRID_SIZE = 6         # 6x6
CELL_SIZE = 100       # taille d'une case en pixels
WIDTH = HEIGHT = GRID_SIZE * CELL_SIZE

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

def listtotuple(l):
    return l[0], l[1]

#Gestion de la q-table
class q_table:
    #Init la q-table avec des valeurs random
    def __init__(self, states, actions):
        #Dico de la forme {((x,y),action):q_value}
        self.table = {}
        self.actions = actions

        for state in states:
            for action in actions:
                self.table[(listtotuple(state), action)] = 0.0

    #Renvoie le prochain état selon la politique epsilon greedy
    def next_move(self, state, actions, epsilon):
        x = random.choices([0,1], weights=[1-epsilon, epsilon])[0]

        #Prendre le plus grand Q
        if x == 0:
            best = self.table[(listtotuple(state), actions[0])]
            best_action = actions[0]

            for action in actions:
                temp = self.table[(listtotuple(state), action)]
                if temp > best:
                    best = temp
                    best_action = action
            return best_action
        else:
            #print("proc")
            x = random.randint(0, len(actions)-1)
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
        for _ in range(random.randint(5, 8)):
            x, y = random.randint(0, self.size-1), random.randint(0, self.size-1)
            if [x, y] not in ([0, 0], [self.size-1, self.size-1]):
                self.waters.append([x, y])
        self.goal = [self.size-1, self.size-1]

    def is_water(self):
        if [self.agent_pos[0],self.agent_pos[1]] in self.waters:
            self.agent_pos = [0,0]
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

#plays a game to learn
def learn(qtable,env, alpha, gamma, epsilon):
    steps = 0
    res = ""
    history = [[0,0]]
    explored = [(0,0)]

    while res != "win" and steps <= 100:
        possible_moves = env.get_possible_moves(env.agent_pos)

        # Garde l'état avant de jouer en mémoire
        s = env.agent_pos

        # Calcule et joue le prochain coup
        next_move = qtable.next_move(env.agent_pos, possible_moves, epsilon)
        res = env.move(next_move)
        steps += 1

        history.append([env.agent_pos[0], env.agent_pos[1]])

        # Choppe le reward
        if res == "win":
            reward = 5
            #print("win")
        elif res == "water":
            reward = -10
        elif steps == 100:
            reward = -5
        elif history[steps-2] == env.agent_pos:
            reward = -8
        else:
            if tuple(env.agent_pos) not in explored:
                reward = 2
                explored.append(tuple(env.agent_pos))
            else:
                reward = 0

        # Donne le meilleur q_val possible à partir de s'
        current_q = qtable.table[(listtotuple(s), next_move)]
        bestfuture_qval = max([qtable.table[(listtotuple(env.agent_pos), a_prime)] for a_prime in env.get_possible_moves(env.agent_pos)])
        qtable.table[(listtotuple(s), next_move)] = current_q + alpha * (reward + (gamma * bestfuture_qval) - current_q)


# --- Boucle principale ---
env = Environment(GRID_SIZE)
running = True

s = [x for x in range(env.size)]
qtable = q_table(list(itertools.product(s,s)), ["UP", "DOWN", "LEFT", "RIGHT"])
alpha = 0.3
gamma = 0.2
epsilon = 1.0

#Plays 1000 games to learn
for i in range(0, 100000):
    learn(qtable, env, alpha, gamma, epsilon)
    epsilon = 0.01 + (1-0.001)*(math.exp(-0.0001 * i))
    #print(epsilon)
env.reset()
print([x for x in qtable.table.items() if x[0][0] == (0,0)])
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        #Détermine la prochaine action
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                possible_moves = env.get_possible_moves(env.agent_pos)

                #Garde l'état avant de jouer en mémoire
                s = env.agent_pos

                #Calcule et joue le prochain coup
                next_move = qtable.next_move(env.agent_pos, possible_moves, epsilon)
                env.move(next_move)
                print([x for x in qtable.table.items() if x[0][0] == tuple(env.agent_pos)])
                #Choppe le reward
                if env.agent_pos == [5,5]:
                    reward = 1
                elif env.agent_pos == [0,0]:
                    reward = -1
                else:
                    reward = 0

                #Donne le meilleur q_val possible à partir de s'
                #best_future_qval = max([qtable.table[(listtotuple(env.agent_pos), a_prime)] for a_prime in env.get_possible_moves(env.agent_pos)])
                #qtable.table[(listtotuple(s), next_move)] = qtable.table[(listtotuple(s), next_move)] + alpha*(reward + gamma*best_future_qval - qtable.table[(listtotuple(s), next_move)])

    env.draw(screen)
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
