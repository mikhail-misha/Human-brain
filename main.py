import math
import turtle
import random
import threading

from celegans import (
    N, NEURONS, CHEM_EDGES, GAP_EDGES,
    TAU, DT, V_REST, V_RESET, V_THRESH, V_MAX, T_REF,
    W_SCALE, noise_amp,
)

running = [True]


def stdin_watcher():
    while running[0]:
        try:
            line = input()
        except EOFError:
            break
        if line.strip().lower() == "stop":
            running[0] = False
            try:
                turtle.bye()
            except Exception:
                pass
            break


threading.Thread(target=stdin_watcher, daemon=True).start()


# ---- нормализованные веса ----
CHEM_NORM = []
for pre, post, w in CHEM_EDGES:
    w_norm = math.tanh(w / 12.0) * 0.15
    CHEM_NORM.append((pre, post, w_norm))

GAP_W = 0.06

# ---- состояние ----
v = [random.uniform(0.0, 0.02) for _ in range(N)]
I_chem = [0.0] * N
I_gap = [0.0] * N
I_stim = [0.0] * N

# фиксированное смещение на нейрон — даёт разнообразие характеров
I_bg_const = [random.gauss(0.06, 0.10) for _ in range(N)]

step_count = 0

TAU_MS = 20.0
DT_MS = 2.0
NOISE_STD = 0.03


def brain_step():
    global v, I_chem, I_gap, I_stim, step_count

    for i in range(N):
        I_chem[i] = 0.0
        I_gap[i] = 0.0

    for pre, post, w in CHEM_NORM:
        I_chem[post] += w * v[pre]

    for a, b, w in GAP_EDGES:
        diff = v[a] - v[b]
        I_gap[a] -= GAP_W * diff
        I_gap[b] += GAP_W * diff

    for i in range(N):
        bg = I_bg_const[i] + random.gauss(0.0, NOISE_STD)
        dv = (-v[i] + I_chem[i] + I_gap[i] + bg + I_stim[i]) / TAU_MS
        v[i] += dv * DT_MS

    # аварийный сброс
    bad = False
    for i in range(N):
        if v[i] != v[i] or abs(v[i]) > 2.0:
            bad = True
            break
    if bad:
        for i in range(N):
            val = v[i]
            if val != val:
                val = 0.0
            if val > 0.3:
                val = 0.3
            if val < -0.3:
                val = -0.3
            v[i] = val

    for i in range(N):
        I_stim[i] *= 0.7

    step_count += 1
    if step_count % 30 == 0:
        print(f"min={min(v):.4f} max={max(v):.4f} "
              f"avg={sum(v)/N:.4f} "
              f"absavg={sum(abs(x) for x in v)/N:.4f}")


# ---- геометрия ----
DOT_PX = 10
GAP_PX = 5
CELL = DOT_PX + GAP_PX

COLS = int(math.ceil(math.sqrt(N)))
ROWS = int(math.ceil(N / COLS))

GRID_W = COLS * CELL
GRID_H = ROWS * CELL

MARGIN = 30
WINDOW_W = max(GRID_W, GRID_H) + 2 * MARGIN
WINDOW_H = WINDOW_W

FIXED_SIZE = DOT_PX / 20.0

screen = turtle.Screen()
screen.setup(width=WINDOW_W, height=WINDOW_H)
screen.bgcolor("black")
screen.title("C. elegans")
screen.colormode(255)
screen.tracer(0)

X0 = -GRID_W / 2 + CELL / 2
Y0 =  GRID_H / 2 - CELL / 2

dots = []
POS = []
for i in range(N):
    row = i // COLS
    col = i % COLS
    x = X0 + col * CELL
    y = Y0 - row * CELL
    POS.append((x, y))

    t = turtle.Turtle()
    t.hideturtle()
    t.speed(0)
    t.penup()
    t.shape("circle")
    t.shapesize(FIXED_SIZE)
    t.color(0, 0, 80)
    t.goto(x, y)
    t.showturtle()
    dots.append(t)


STOPS = [
    (0.00, (10,  15,  40)),
    (0.10, (15,  25,  70)),
    (0.25, (25,  50,  120)),
    (0.45, (40,  90,  180)),
    (0.65, (70,  140, 230)),
    (0.85, (110, 180, 255)),
    (1.00, (160, 210, 255)),
]


def activation_color(a):
    a = max(0.0, min(1.0, a))
    for i in range(len(STOPS) - 1):
        a0, c0 = STOPS[i]
        a1, c1 = STOPS[i + 1]
        if a0 <= a <= a1:
            t = 0.0 if a1 == a0 else (a - a0) / (a1 - a0)
            r = int(c0[0] + (c1[0] - c0[0]) * t)
            g = int(c0[1] + (c1[1] - c0[1]) * t)
            b = int(c0[2] + (c1[2] - c0[2]) * t)
            return (r, g, b)
    return STOPS[-1][1]


last_color = [None] * N


def on_click(x, y):
    best_i = -1
    best_d2 = 1e18
    for i in range(N):
        px, py = POS[i]
        d2 = (px - x) ** 2 + (py - y) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i
    if best_i >= 0 and best_d2 <= 12 * 12:
        I_stim[best_i] += 0.8
        print(f"STIM {NEURONS[best_i]} [{best_i}]")


screen.onclick(on_click)


def loop():
    if not running[0]:
        return

    try:
        brain_step()

        sorted_v = sorted(v)
        median = sorted_v[N // 2]
        q1 = sorted_v[N // 4]
        q3 = sorted_v[3 * N // 4]
        iqr = q3 - q1
        if iqr < 1e-6:
            iqr = 1e-6

        for i in range(N):
            z = (v[i] - median) / iqr
            a = 0.5 + z * 0.5
            a = max(0.0, min(1.0, a))
            a_r = round(a, 2)

            if a_r != last_color[i]:
                r, g, b = activation_color(a_r)
                dots[i].color(r, g, b)
                last_color[i] = a_r

        screen.update()
    except turtle.Terminator:
        running[0] = False
        return
    except Exception:
        running[0] = False
        return

    screen.ontimer(loop, 60)


try:
    screen.ontimer(loop, 60)
    screen.mainloop()
except KeyboardInterrupt:
    pass
finally:
    running[0] = False
    try:
        turtle.bye()
    except Exception:
        pass
