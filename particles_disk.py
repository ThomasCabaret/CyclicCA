import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use Tkinter backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
import os

"""
2D Particle Simulation with Buckets, Speed-based Color, and Adaptive Radius
---------------------------------------------------------------------------
Features:
1) Particles in a unit square (0..1 x 0..1).
2) Bucket-based collision detection:
   - We divide the domain into square cells (buckets) of size ~ (3 * radius).
   - Each particle is placed in exactly one bucket.
   - Collisions are checked only among particles in the same bucket or in
     the 8 neighboring buckets.
3) Particle color depends on speed, from black (0 speed) to red (mid speed) to yellow (max speed).
4) The radius of each particle adapts to the number of particles:
   - We fix the total area fraction (e.g. 25% of the domain).
   - radius = sqrt( fraction_area / (pi * N) ).
5) Same interface:
   - Play/Pause
   - Speed +/-
   - Reset / Init
   - Record frames
   - Number of particles
   - Particles have elastic collisions with walls and with each other.
"""

# ----------------- Global Defaults -----------------
IMG_DIRECTORY = "frames_particles"  # Where to save frames
DEFAULT_NUM_PARTICLES = 10          # Default number of particles
DEFAULT_SPEED = 10                  # Minimal delay in ms => "max speed" of update
MAX_INITIAL_VEL = 0.3              # Random velocities are in [-MAX_INITIAL_VEL, +MAX_INITIAL_VEL]
ELASTICITY = 1.0                    # Perfectly elastic collisions
FRACTION_AREA = 0.25                # Fraction of domain area occupied by all particles' areas (max)

# Frame counter for recording
frame_counter = 0

class ParticleSimulationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("2D Particle Simulation (Buckets, Speed Color, Adaptive Radius)")
        self.root.resizable(True, True)

        # Simulation state
        self.running = False        # Whether the simulation is running
        self.record = False         # Whether we record frames
        self.speed = DEFAULT_SPEED  # The "after" interval in ms (lower => faster updates)

        # Number of particles (user-adjustable)
        self.num_particles_var = tk.IntVar(value=DEFAULT_NUM_PARTICLES)

        # Create the frames directory if it doesn't exist
        if not os.path.exists(IMG_DIRECTORY):
            os.makedirs(IMG_DIRECTORY)

        # --- Figure & Axes ---
        self.fig, self.ax = plt.subplots(figsize=(6, 6))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.set_aspect("equal", "box")
        self.ax.set_title("Particle Simulation with Buckets")
        self.ax.axis("on")

        # Embed Matplotlib in Tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=10, sticky="nsew")

        # Particle data:
        # positions: Nx2 array of (x, y)
        # velocities: Nx2 array of (vx, vy)
        # patches: list of circle artists
        self.positions = None
        self.velocities = None
        self.patches = []

        # "Initial" copies for the Init button
        self.initial_positions = None
        self.initial_velocities = None

        # The radius is computed based on the number of particles
        # so that their total area is fraction_area of the unit square.
        self.radius = None

        # Initialize with default number of particles
        self.init_new_particles(self.num_particles_var.get())

        # --- Control Panel ---
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Play / Pause
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.start_simulation)
        self.play_button.grid(row=0, column=0, pady=5, sticky="ew")

        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_simulation)
        self.pause_button.grid(row=1, column=0, pady=5, sticky="ew")

        # Reset / Init
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset_particles)
        self.reset_button.grid(row=2, column=0, pady=5, sticky="ew")

        self.init_button = ttk.Button(self.control_frame, text="Init", command=self.init_particles_from_saved)
        self.init_button.grid(row=3, column=0, pady=5, sticky="ew")

        # Speed
        self.speed_up_button = ttk.Button(self.control_frame, text="Speed +", command=self.speed_up)
        self.speed_up_button.grid(row=4, column=0, pady=5, sticky="ew")

        self.speed_down_button = ttk.Button(self.control_frame, text="Speed -", command=self.speed_down)
        self.speed_down_button.grid(row=5, column=0, pady=5, sticky="ew")

        # Record
        self.record_var = tk.BooleanVar(value=False)
        self.record_check = ttk.Checkbutton(
            self.control_frame,
            text="Record",
            variable=self.record_var,
            command=self.toggle_record
        )
        self.record_check.grid(row=6, column=0, pady=5, sticky="ew")

        # Number of particles
        ttk.Label(self.control_frame, text="#Particles").grid(row=7, column=0, pady=(5, 0), sticky="ew")
        self.num_particles_spin = ttk.Spinbox(
            self.control_frame,
            from_=1, to=200,
            textvariable=self.num_particles_var,
            width=5
        )
        self.num_particles_spin.grid(row=8, column=0, pady=5, sticky="ew")

        # Configure resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Window close handling
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start the update loop
        self.update_loop()

    # ----------------------------------------------------------------
    #                 Particle Initialization / Reset
    # ----------------------------------------------------------------
    def init_new_particles(self, n):
        """
        Create n new random particles in the unit square.
        Each has a random position and velocity.
        The radius is adapted so that the total area of all particles
        is FRACTION_AREA of the unit square.
        """
        # Compute radius to keep total area below FRACTION_AREA
        # domain area = 1
        # total area of n circles = n * pi * r^2 <= FRACTION_AREA
        # => r^2 = FRACTION_AREA / (n * pi)
        # => r = sqrt(FRACTION_AREA / (n * pi))
        self.radius = np.sqrt(FRACTION_AREA / (n * np.pi))

        self.positions = np.zeros((n, 2), dtype=float)
        self.velocities = np.zeros((n, 2), dtype=float)

        # Random positions in [radius, 1-radius]
        for i in range(n):
            x = np.random.uniform(self.radius, 1.0 - self.radius)
            y = np.random.uniform(self.radius, 1.0 - self.radius)
            self.positions[i] = [x, y]

            vx = np.random.uniform(-MAX_INITIAL_VEL, MAX_INITIAL_VEL)
            vy = np.random.uniform(-MAX_INITIAL_VEL, MAX_INITIAL_VEL)
            self.velocities[i] = [vx, vy]

        # Remove old patches
        for c in self.patches:
            c.remove()
        self.patches.clear()

        # Create circle patches
        for i in range(n):
            circle = plt.Circle(
                (self.positions[i, 0], self.positions[i, 1]),
                radius=self.radius,
                fc="blue",
                ec="black",
                alpha=0.8
            )
            self.ax.add_patch(circle)
            self.patches.append(circle)

        # Save initial copies
        self.initial_positions = self.positions.copy()
        self.initial_velocities = self.velocities.copy()

        # Redraw
        self.canvas.draw()

    def reset_particles(self):
        """
        Generate a new random set of particles, using the value in num_particles_var.
        """
        n = self.num_particles_var.get()
        self.init_new_particles(n)

    def init_particles_from_saved(self):
        """
        Revert to the last 'initial' positions/velocities.
        """
        if self.initial_positions is None or self.initial_velocities is None:
            return
        self.positions = self.initial_positions.copy()
        self.velocities = self.initial_velocities.copy()

        # Move circles accordingly
        for i, circle in enumerate(self.patches):
            circle.center = (self.positions[i, 0], self.positions[i, 1])
        self.canvas.draw()

    # ----------------------------------------------------------------
    #                            GUI Actions
    # ----------------------------------------------------------------
    def start_simulation(self):
        self.running = True

    def pause_simulation(self):
        self.running = False

    def speed_up(self):
        """Decrease the update interval => faster (minimum 1 ms)."""
        self.speed = max(1, self.speed - 10)

    def speed_down(self):
        """Increase the update interval => slower."""
        self.speed += 10

    def toggle_record(self):
        self.record = self.record_var.get()

    # ----------------------------------------------------------------
    #                       Main Loop
    # ----------------------------------------------------------------
    def update_loop(self):
        """Periodically update the simulation if running."""
        if not self.root.winfo_exists():
            return  # Window closed

        if self.running:
            self.update_simulation()
            if self.record:
                self.save_frame()

        self.root.after(self.speed, self.update_loop)

    # ----------------------------------------------------------------
    #                       Simulation Update
    # ----------------------------------------------------------------
    def update_simulation(self):
        """
        Move particles, handle wall collisions, handle particle collisions (via buckets),
        and update their colors based on speed.
        """
        dt = 0.01
        n = len(self.positions)

        # 1) Move
        self.positions += self.velocities * dt

        # 2) Collisions with walls
        for i in range(n):
            x, y = self.positions[i]
            vx, vy = self.velocities[i]

            # If out of left or right boundary
            if x < self.radius:
                x = self.radius
                vx = abs(vx) * ELASTICITY
            elif x > 1.0 - self.radius:
                x = 1.0 - self.radius
                vx = -abs(vx) * ELASTICITY

            # If out of bottom or top boundary
            if y < self.radius:
                y = self.radius
                vy = abs(vy) * ELASTICITY
            elif y > 1.0 - self.radius:
                y = 1.0 - self.radius
                vy = -abs(vy) * ELASTICITY

            self.positions[i] = [x, y]
            self.velocities[i] = [vx, vy]

        # 3) Particle-particle collisions using buckets
        self.bucket_collision_check()

        # 4) Update patches (positions + facecolor based on speed)
        speeds = np.sqrt(np.sum(self.velocities**2, axis=1))
        max_speed = speeds.max() if len(speeds) > 0 else 1e-9

        for i, circle in enumerate(self.patches):
            # Update center
            circle.center = (self.positions[i, 0], self.positions[i, 1])

            # Set color based on speed
            speed_ratio = 0.0
            if max_speed > 1e-9:
                speed_ratio = speeds[i] / max_speed
            # speed_ratio in [0..1], map black->red->yellow
            color = self.speed_to_color(speed_ratio)
            circle.set_facecolor(color)

        self.canvas.draw()

    def bucket_collision_check(self):
        """
        Build a grid of buckets. Each bucket covers a cell of side ~ 3*radius.
        Particles in the same or adjacent buckets are checked for collisions.
        """
        n = len(self.positions)
        if n <= 1:
            return

        cell_size = 3.0 * self.radius
        # number of cells along x or y
        nx = int(np.ceil(1.0 / cell_size))
        ny = int(np.ceil(1.0 / cell_size))

        # Buckets: a dictionary keyed by (ix, iy) with list of particle indices
        buckets = {}
        for i in range(nx):
            for j in range(ny):
                buckets[(i, j)] = []

        # Assign each particle to a bucket
        for i in range(n):
            x, y = self.positions[i]
            ix = int(x // cell_size)
            iy = int(y // cell_size)
            # clamp index in [0, nx-1], [0, ny-1] just in case
            ix = min(nx-1, max(0, ix))
            iy = min(ny-1, max(0, iy))
            buckets[(ix, iy)].append(i)

        # For each bucket, check collisions among particles in the same bucket
        # plus the 8 neighboring buckets
        bucket_neighbors = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),  (0, 0),  (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

        for ix in range(nx):
            for iy in range(ny):
                indices_in_bucket = []
                # Gather all particle indices from neighbors
                for dx, dy in bucket_neighbors:
                    bx = ix + dx
                    by = iy + dy
                    if 0 <= bx < nx and 0 <= by < ny:
                        indices_in_bucket.extend(buckets[(bx, by)])

                # Now check collisions among these indices
                self.check_collisions_in_list(indices_in_bucket)

    def check_collisions_in_list(self, indices):
        """
        Check collisions (elastic) among all particle indices in the given list.
        """
        ln = len(indices)
        for i_idx in range(ln):
            i = indices[i_idx]
            for j_idx in range(i_idx+1, ln):
                j = indices[j_idx]

                dx = self.positions[j, 0] - self.positions[i, 0]
                dy = self.positions[j, 1] - self.positions[i, 1]
                dist_sq = dx*dx + dy*dy
                min_dist = 2.0 * self.radius

                if dist_sq < (min_dist * min_dist):
                    # collision
                    dist = np.sqrt(dist_sq)
                    if dist < 1e-12:
                        # Rare exact overlap => separate them artificially
                        dist = min_dist
                        dx = min_dist
                        dy = 0.0

                    nx = dx / dist
                    ny = dy / dist

                    # Relative velocity
                    dvx = self.velocities[j, 0] - self.velocities[i, 0]
                    dvy = self.velocities[j, 1] - self.velocities[i, 1]
                    rel_vel = dvx * nx + dvy * ny

                    # Are they moving toward each other?
                    if rel_vel < 0:
                        # 1D elastic collision for same mass
                        impulse = -(1 + ELASTICITY) * rel_vel / 2.0
                        ix = impulse * nx
                        iy = impulse * ny

                        self.velocities[i, 0] -= ix
                        self.velocities[i, 1] -= iy
                        self.velocities[j, 0] += ix
                        self.velocities[j, 1] += iy

                    # minimal position fix
                    overlap = min_dist - dist
                    self.positions[i, 0] -= 0.5 * overlap * nx
                    self.positions[i, 1] -= 0.5 * overlap * ny
                    self.positions[j, 0] += 0.5 * overlap * nx
                    self.positions[j, 1] += 0.5 * overlap * ny

    # ----------------------------------------------------------------
    #                     Speed-to-Color Mapping
    # ----------------------------------------------------------------
    def speed_to_color(self, ratio):
        """
        Given a ratio in [0..1], interpolate from black->red->yellow.
        - ratio=0 => black (0,0,0)
        - ratio=0.5 => red   (1,0,0)
        - ratio=1 => yellow (1,1,0)
        We'll do a piecewise linear interpolation:
           0   to 0.5 => black->red
           0.5 to 1.0 => red->yellow
        """
        if ratio <= 0.5:
            # black->red
            t = ratio / 0.5  # in [0..1]
            r = t
            g = 0
            b = 0
        else:
            # red->yellow
            t = (ratio - 0.5) / 0.5  # in [0..1]
            r = 1
            g = t
            b = 0

        return (r, g, b)

    # ----------------------------------------------------------------
    #                           Recording
    # ----------------------------------------------------------------
    def save_frame(self):
        """
        Save the current frame as a PNG.
        """
        global frame_counter
        frame_name = f"{IMG_DIRECTORY}/frame_{frame_counter:04d}.png"

        self.fig.canvas.draw()
        img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
        w, h = self.fig.canvas.get_width_height()
        img = img.reshape((h, w, 3))
        pil_img = Image.fromarray(img)
        pil_img.save(frame_name)

        frame_counter += 1

    # ----------------------------------------------------------------
    #                        Window Close
    # ----------------------------------------------------------------
    def on_close(self):
        """Stop simulation and close the window."""
        self.running = False
        self.root.destroy()

# -------------------------------------------------------------------
#                                MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ParticleSimulationApp(root)
    root.mainloop()
