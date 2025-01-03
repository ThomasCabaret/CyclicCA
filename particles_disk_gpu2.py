import sys
import math
import numpy as np
from numba import cuda
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *

WIN_WIDTH = 1280
WIN_HEIGHT = 720
PARTICLE_COUNT = 2000
PARTICLE_RADIUS = 3.0
BUCKET_SIZE = 3.0 * PARTICLE_RADIUS
WORLD_SIZE = 2000.0
THREADS_PER_BLOCK = 256

@cuda.jit
def phase1_update_collision(px, py, vx, vy, npx, npy, nvx, nvy, radius, count, world_size):
    i = cuda.grid(1)
    if i >= count:
        return

    dt = 0.016
    x_i = px[i]
    y_i = py[i]
    vx_i = vx[i]
    vy_i = vy[i]

    # Update position with velocity
    x_new = x_i + vx_i * dt
    y_new = y_i + vy_i * dt

    # Boundary collision
    if x_new < 0.0 or x_new > world_size:
        vx_i = -vx_i
        x_new = min(max(x_new, 0.0), world_size)
    if y_new < 0.0 or y_new > world_size:
        vy_i = -vy_i
        y_new = min(max(y_new, 0.0), world_size)

    # Check collisions with other particles
    for j in range(count):
        if j == i:
            continue

        dx = px[j] - x_new
        dy = py[j] - y_new
        dist_sq = dx * dx + dy * dy
        min_dist = 2.0 * radius

        if dist_sq < min_dist * min_dist:
            dist = math.sqrt(dist_sq)
            if dist < 1e-12:
                dist = min_dist
                dx = min_dist
                dy = 0.0

            # Normal vector
            nx = dx / dist
            ny = dy / dist

            # Relative velocity
            dvx = vx[j] - vx_i
            dvy = vy[j] - vy_i
            rel_vel = dvx * nx + dvy * ny

            # Only resolve if particles are moving toward each other
            if rel_vel < 0:
                impulse = -(1.0 + 1.0) * rel_vel / 2.0
                ix = impulse * nx
                iy = impulse * ny

                # Update velocities symmetrically
                vx_i -= ix
                vy_i -= iy
                vx[j] += ix
                vy[j] += iy

            # Minimal position correction to resolve overlap
            overlap = min_dist - dist
            x_new -= 0.5 * overlap * nx
            y_new -= 0.5 * overlap * ny
            px[j] += 0.5 * overlap * nx
            py[j] += 0.5 * overlap * ny

    # Write back new positions and velocities
    npx[i] = x_new
    npy[i] = y_new
    nvx[i] = vx_i
    nvy[i] = vy_i

@cuda.jit
def phase2_reassign_buckets(bucket_array, grid_size):
    i = cuda.grid(1)
    if i >= grid_size:
        return
    bucket_array[i, 0] = 0

@cuda.jit
def fill_buckets(px, py, bucket_size, bucket_array,
                 radius, world_size, count, grid_dim):
    i = cuda.grid(1)
    if i >= count:
        return
    x = px[i]
    y = py[i]
    gx = int(x // bucket_size)
    gy = int(y // bucket_size)
    if gx < 0: gx = 0
    if gy < 0: gy = 0
    if gx >= grid_dim: gx = grid_dim - 1
    if gy >= grid_dim: gy = grid_dim - 1
    idx = gy * grid_dim + gx
    offset = cuda.atomic.add(bucket_array[idx], 0, 1)
    bucket_array[idx, offset + 1] = i

def color_from_speed(s):
    c = max(min(s / 50.0, 1.0), 0.0)
    if c <= 0.5:
        t = c / 0.5
        r = t
        g = 0.0
        b = 1.0 - t
    else:
        t = (c - 0.5) / 0.5
        r = 1.0
        g = t
        b = 0.0
    return (r, g, b)

def draw_disk(x, y, r, slices, cr, cg, cb):
    glColor3f(cr, cg, cb)
    glBegin(GL_TRIANGLE_FAN)
    glVertex2f(x, y)
    step = 2.0 * math.pi / slices
    a = 0.0
    for _ in range(slices + 1):
        cx = x + r * math.cos(a)
        cy = y + r * math.sin(a)
        glVertex2f(cx, cy)
        a += step
    glEnd()

def main():
    pygame.init()
    pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT), OPENGL | DOUBLEBUF | RESIZABLE)

    offset_x = 0.0
    offset_y = 0.0
    scale = 1.0
    aspect = float(WIN_WIDTH) / float(WIN_HEIGHT)

    def update_view(cx=None, cy=None, factor=None):
        nonlocal offset_x, offset_y, scale, aspect
        if cx is not None and cy is not None and factor is not None:
            scale *= factor
            offset_x = cx - (WORLD_SIZE / scale) * 0.5
            offset_y = cy - (WORLD_SIZE / scale / aspect) * 0.5

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glViewport(0, 0, WIN_WIDTH, WIN_HEIGHT)
        gluOrtho2D(offset_x, offset_x + (WORLD_SIZE / scale),
                   offset_y, offset_y + (WORLD_SIZE / scale / aspect))
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

    update_view()

    dragging = False
    drag_start_x = 0
    drag_start_y = 0
    offset_start_x = 0.0
    offset_start_y = 0.0

    pos_x = np.random.rand(PARTICLE_COUNT).astype(np.float32) * WORLD_SIZE
    pos_y = np.random.rand(PARTICLE_COUNT).astype(np.float32) * WORLD_SIZE
    vel_x = (np.random.rand(PARTICLE_COUNT).astype(np.float32) - 0.5) * 200.0
    vel_y = (np.random.rand(PARTICLE_COUNT).astype(np.float32) - 0.5) * 200.0
    new_pos_x = np.zeros_like(pos_x)
    new_pos_y = np.zeros_like(pos_y)
    new_vel_x = np.zeros_like(vel_x)
    new_vel_y = np.zeros_like(vel_y)

    d_pos_x = cuda.to_device(pos_x)
    d_pos_y = cuda.to_device(pos_y)
    d_vel_x = cuda.to_device(vel_x)
    d_vel_y = cuda.to_device(vel_y)
    d_new_pos_x = cuda.to_device(new_pos_x)
    d_new_pos_y = cuda.to_device(new_pos_y)
    d_new_vel_x = cuda.to_device(new_vel_x)
    d_new_vel_y = cuda.to_device(new_vel_y)

    grid_dim = int(math.ceil(WORLD_SIZE / BUCKET_SIZE))
    bucket_array = np.zeros((grid_dim * grid_dim, PARTICLE_COUNT + 1), dtype=np.int32)
    d_bucket_array = cuda.to_device(bucket_array)

    blocks_p = (PARTICLE_COUNT + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
    blocks_b = (grid_dim * grid_dim + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                running = False

            elif event.type == VIDEORESIZE:
                WIN_W, WIN_H = event.w, event.h
                if WIN_W < 1: WIN_W = 1
                if WIN_H < 1: WIN_H = 1
                # Keep local copies up to date
                aspect = float(WIN_W) / float(WIN_H)
                pygame.display.set_mode((WIN_W, WIN_H), OPENGL | DOUBLEBUF | RESIZABLE)
                update_view()

            elif event.type == MOUSEWHEEL:
                center_x = offset_x + (WORLD_SIZE / scale) * 0.5
                center_y = offset_y + (WORLD_SIZE / scale / aspect) * 0.5
                if event.y > 0:
                    update_view(cx=center_x, cy=center_y, factor=1.1)
                else:
                    update_view(cx=center_x, cy=center_y, factor=1.0 / 1.1)

            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:
                    dragging = True
                    drag_start_x, drag_start_y = event.pos
                    offset_start_x = offset_x
                    offset_start_y = offset_y

            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False

            elif event.type == MOUSEMOTION and dragging:
                mx, my = event.pos
                dx = mx - drag_start_x
                dy = my - drag_start_y
                world_drag_x = (dx / WIN_WIDTH) * (WORLD_SIZE / scale)
                world_drag_y = (dy / WIN_HEIGHT) * (WORLD_SIZE / scale / aspect)
                offset_x = offset_start_x - world_drag_x
                offset_y = offset_start_y + world_drag_y
                update_view()

            elif event.type == KEYDOWN:
                if event.key == K_UP:
                    mx, my = pygame.mouse.get_pos()
                    wcx = offset_x + (mx / WIN_WIDTH) * (WORLD_SIZE / scale)
                    wcy = offset_y + ((WIN_HEIGHT - my) / WIN_HEIGHT) * (WORLD_SIZE / scale / aspect)
                    update_view(cx=wcx, cy=wcy, factor=1.1)
                elif event.key == K_DOWN:
                    mx, my = pygame.mouse.get_pos()
                    wcx = offset_x + (mx / WIN_WIDTH) * (WORLD_SIZE / scale)
                    wcy = offset_y + ((WIN_HEIGHT - my) / WIN_HEIGHT) * (WORLD_SIZE / scale / aspect)
                    update_view(cx=wcx, cy=wcy, factor=1.0 / 1.1)

        phase1_update_collision[blocks_p, THREADS_PER_BLOCK](
            d_pos_x, d_pos_y, d_vel_x, d_vel_y,
            d_new_pos_x, d_new_pos_y, d_new_vel_x, d_new_vel_y,
            PARTICLE_RADIUS, PARTICLE_COUNT, WORLD_SIZE
        )
        cuda.synchronize()

        tmp_pos_x = d_pos_x
        tmp_pos_y = d_pos_y
        tmp_vel_x = d_vel_x
        tmp_vel_y = d_vel_y
        d_pos_x = d_new_pos_x
        d_pos_y = d_new_pos_y
        d_vel_x = d_new_vel_x
        d_vel_y = d_new_vel_y
        d_new_pos_x = tmp_pos_x
        d_new_pos_y = tmp_pos_y
        d_new_vel_x = tmp_vel_x
        d_new_vel_y = tmp_vel_y
        cuda.synchronize()

        phase2_reassign_buckets[blocks_b, THREADS_PER_BLOCK](
            d_bucket_array, grid_dim * grid_dim
        )
        cuda.synchronize()

        fill_buckets[blocks_p, THREADS_PER_BLOCK](
            d_pos_x, d_pos_y, BUCKET_SIZE,
            d_bucket_array, PARTICLE_RADIUS, WORLD_SIZE,
            PARTICLE_COUNT, grid_dim
        )
        cuda.synchronize()

        glClear(GL_COLOR_BUFFER_BIT)
        host_x = d_pos_x.copy_to_host()
        host_y = d_pos_y.copy_to_host()
        host_vx = d_vel_x.copy_to_host()
        host_vy = d_vel_y.copy_to_host()

        for i in range(PARTICLE_COUNT):
            s = math.sqrt(host_vx[i] * host_vx[i] + host_vy[i] * host_vy[i])
            r, g, b = color_from_speed(s)
            draw_disk(host_x[i], host_y[i], PARTICLE_RADIUS, 12, r, g, b)

        pygame.display.flip()
        pygame.time.wait(10)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
