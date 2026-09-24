import numpy as np
import pygame
import sys
import os
import math
from brian2 import *

prefs.codegen.target = 'numpy'

pygame.init()

info = pygame.display.Info()
SCREEN_W = info.current_w
SCREEN_H = info.current_h

WINDOW_W = min(SCREEN_W - 40, 1366)
WINDOW_H = min(SCREEN_H - 80, 768)

if WINDOW_W < 400:
    WINDOW_W = 400
if WINDOW_H < 300:
    WINDOW_H = 300

os.environ['SDL_VIDEO_CENTERED'] = '1'
screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
pygame.display.set_caption("H01 — симуляция коры")
clock = pygame.time.Clock()

print("Загрузка коннектома H01")

data = np.load("human_edges.npz", allow_pickle=True)
edges = data['edges']

unique_ids = np.unique(np.concatenate([edges[:, 0], edges[:, 1]]))
N = len(unique_ids)

print(f"Нейронов: {N}")
print(f"Связей: {len(edges)}")

id_to_local = {old_id: new_id for new_id, old_id in enumerate(unique_ids)}

chem_matrix = np.zeros((N, N), dtype=np.float32)
gap_matrix = np.zeros((N, N), dtype=np.float32)

chem_count = 0
gap_count = 0

for pre, post, syn_type in edges:
    local_pre = id_to_local[pre]
    local_post = id_to_local[post]

    if syn_type == 1:
        chem_matrix[local_post, local_pre] += 1.0
        chem_count += 1
    elif syn_type == 2:
        gap_matrix[local_post, local_pre] += 1.0
        gap_matrix[local_pre, local_post] += 1.0
        gap_count += 1

print(f"Химических синапсов: {chem_count}")
print(f"Электрических контактов: {gap_count}")

print("Инициализация сети")

start_scope()

eqs = '''
dv/dt = (-v + I_chem + I_gap + I_bg) / (20*ms) : 1
I_chem : 1
I_gap : 1
I_bg : 1
'''

neurons = NeuronGroup(N, eqs, method='exponential_euler')
neurons.v = 'rand() * 0.02'
neurons.I_chem = 0
neurons.I_gap = 0
neurons.I_bg = 0

chem_conn = np.argwhere(chem_matrix > 0)
if len(chem_conn) > 0:
    syn_chem = Synapses(neurons, neurons, model='''
        w : 1
        I_chem_post = w * v_pre : 1 (summed)
    ''')
    syn_chem.connect(i=chem_conn[:, 0], j=chem_conn[:, 1])
    chem_weights = chem_matrix[chem_conn[:, 0], chem_conn[:, 1]]
    syn_chem.w = np.tanh(chem_weights / 3.0) * 0.15
    print(f"Химические синапсы подключены: {len(chem_conn)}")
else:
    syn_chem = None

gap_conn = np.argwhere(gap_matrix > 0)
if len(gap_conn) > 0:
    syn_gap = Synapses(neurons, neurons, model='''
        w : 1
        I_gap_post = w * (v_pre - v_post) : 1 (summed)
    ''')
    syn_gap.connect(i=gap_conn[:, 0], j=gap_conn[:, 1])
    gap_weights = gap_matrix[gap_conn[:, 0], gap_conn[:, 1]]
    syn_gap.w = np.tanh(gap_weights / 3.0) * 0.08
    print(f"Электрические контакты подключены: {len(gap_conn)}")
else:
    syn_gap = None

if syn_chem is not None and syn_gap is not None:
    net = Network(neurons, syn_chem, syn_gap)
elif syn_chem is not None:
    net = Network(neurons, syn_chem)
elif syn_gap is not None:
    net = Network(neurons, syn_gap)
else:
    net = Network(neurons)

col_count = int(np.ceil(np.sqrt(N)))
row_count = col_count
while col_count * row_count < N:
    col_count += 1
    row_count = col_count

grid_pad = 20
grid_w = WINDOW_W - grid_pad * 2
grid_h = WINDOW_H - grid_pad * 2
cell = min(grid_w // col_count, grid_h // row_count)
point_radius = max(1, cell // 2 - 1)

if point_radius < 1:
    point_radius = 1

grid_x = grid_pad + (grid_w - col_count * cell) // 2
grid_y = grid_pad + (grid_h - row_count * cell) // 2

positions = []
for i in range(N):
    c = i % col_count
    r = i // col_count
    x = grid_x + c * cell + cell // 2
    y = grid_y + r * cell + cell // 2
    positions.append((x, y))

print(f"Сетка: {col_count} x {row_count}")
print("Симуляция запущена. ESC — выход, SPACE — пауза.")


def v_to_color(volt):
    if volt < 0.02:
        return (5, 5, 15)
    elif volt < 0.05:
        return (10, 15, 40)
    elif volt < 0.10:
        return (15, 25, 70)
    elif volt < 0.20:
        return (25, 50, 120)
    elif volt < 0.35:
        return (40, 90, 180)
    elif volt < 0.55:
        return (70, 140, 230)
    elif volt < 0.85:
        return (110, 180, 255)
    elif volt < 1.20:
        return (160, 210, 255)
    elif volt < 2.0:
        return (200, 230, 255)
    else:
        return (255, 255, 255)


running = True
frame = 0
paused = False
nan_count = 0

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            if event.key == pygame.K_SPACE:
                paused = not paused

    if not paused:
        neurons.I_bg = np.random.randn(N) * 0.05 + 0.08

        net.run(2 * ms)

        v_np = np.asarray(neurons.v).copy()

        if np.any(np.isnan(v_np)) or np.any(np.abs(v_np) > 5.0):
            nan_count += 1
            neurons.v = np.clip(
                np.nan_to_num(v_np, nan=0.0, posinf=2.0, neginf=-2.0),
                -2.0, 2.0
            )
    else:
        v_np = np.asarray(neurons.v).copy()

    screen.fill((0, 0, 0))

    active_count = 0
    high_count = 0
    very_high_count = 0

    for i in range(N):
        volt = v_np[i]
        color = v_to_color(volt)
        pygame.draw.circle(screen, color, positions[i], point_radius)

        if volt > 0.2:
            active_count += 1
        if volt > 0.5:
            high_count += 1
        if volt > 1.0:
            very_high_count += 1

    avg_v = float(np.mean(v_np))
    max_v = float(np.max(v_np))
    min_v = float(np.min(v_np))

    font = pygame.font.Font(None, 18)

    info1 = font.render(
        f"Нейронов: {N}    Активных: {active_count}    "
        f"Ярких: {high_count}    Сверхярких: {very_high_count}",
        True, (200, 200, 200))
    screen.blit(info1, (10, 10))

    info2 = font.render(
        f"Среднее v: {avg_v:.3f}    Мин: {min_v:.3f}    Макс: {max_v:.3f}    "
        f"Сбросов: {nan_count}",
        True, (200, 200, 200))
    screen.blit(info2, (10, 32))

    if paused:
        pause_font = pygame.font.Font(None, 36)
        pause_text = pause_font.render("Пауза", True, (255, 255, 0))
        pause_rect = pause_text.get_rect(center=(WINDOW_W // 2, 70))
        screen.blit(pause_text, pause_rect)

    pygame.display.flip()
    clock.tick(15)
    frame += 1

pygame.quit()
sys.exit()
