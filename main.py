import pygame
import random

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

    def win(self):
        if [self.agent_pos[0], self.agent_pos[1]] == self.goal:
            self.agent_pos = [0, 0]
            print("GG")

    def move(self, action):
        if action == "UP":
            self.agent_pos[1] = max(0, self.agent_pos[1] - 1)
        elif action == "DOWN":
            self.agent_pos[1] = min(self.size - 1, self.agent_pos[1] + 1)
        elif action == "LEFT":
            self.agent_pos[0] = max(0, self.agent_pos[0] - 1)
        elif action == "RIGHT":
            self.agent_pos[0] = min(self.size - 1, self.agent_pos[0] + 1)

        self.is_water()
        self.win()
        
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

# --- Boucle principale ---
env = Environment(GRID_SIZE)
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                env.move("UP")
            elif event.key == pygame.K_DOWN:
                env.move("DOWN")
            elif event.key == pygame.K_LEFT:
                env.move("LEFT")
            elif event.key == pygame.K_RIGHT:
                env.move("RIGHT")
            elif event.key == pygame.K_r:
                env.reset()

    env.draw(screen)
    pygame.display.flip()
    clock.tick(10)

pygame.quit()
