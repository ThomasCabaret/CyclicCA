import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use Tkinter backend
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import os

"""
Cyclic Cellular Automaton (Further Enhanced Version)
----------------------------------------------------
Features:
1) Centralized base color list (black->red->yellow->white).
2) UI controls to adjust:
   - Number of colors in the cycle.
   - Simulation width and height (applied at next reset).
3) Grid updates are modulo, as before.
4) All code and comments in English.

Usage:
- Play: start the simulation
- Pause: stop the simulation
- Reset: create a brand new random grid with new width, height, and number of colors
- Init: revert to the last stored initial grid (from the most recent Reset)
- Speed + / Speed -: adjust simulation speed
- Record checkbox: toggle frame saving in the 'frames' folder
- Neighborhood: choose 4 (von Neumann) or 8 (Moore)
- Threshold spinbox: how many neighbors in (cell_state+1) mod state_count are needed
  for the cell to increment its state.
- #Colors: how many discrete states (and thus color steps) to use in the automaton.
- Grid Width / Grid Height: dimension of the next grid when resetting.
"""

# ----------------- Global Defaults -----------------
DEFAULT_GRID_WIDTH = 300    # Default width
DEFAULT_GRID_HEIGHT = 300   # Default height
DEFAULT_NUM_COLORS = 10     # Default number of colors (states)
DEFAULT_SPEED = 10          # Start with minimal delay (max speed)
IMG_DIRECTORY = "frames"    # Where to save frames

# Global counter for saved frames
frame_counter = 0

# -------------------------------------------------------------------
#                    Colormap & Automaton Functions
# -------------------------------------------------------------------
def create_gradient_cmap(base_colors, n_colors):
    """
    Create a discrete ListedColormap from a list of base colors
    and a total number of discrete color steps n_colors.

    base_colors: list of RGB tuples in [0..1].
    n_colors: how many discrete steps to generate in the colormap.
    """
    # Create a continuous colormap from the base colors
    continuous_cmap = LinearSegmentedColormap.from_list(
        "temp_cmap", base_colors, N=n_colors
    )
    # Sample n_colors discrete values
    color_array = continuous_cmap(np.linspace(0, 1, n_colors))
    # Convert to a ListedColormap
    discrete_cmap = ListedColormap(color_array)
    return discrete_cmap

def init_grid(width, height, states):
    """
    Create a random grid of shape (height, width),
    with values in [0, states-1].
    """
    return np.random.randint(0, states, (height, width))

def update_grid(grid, states, neighborhood=8, threshold=1):
    """
    Update the grid according to a cyclic automaton rule:
    - A cell increments its state (mod 'states') if
      the number of neighbors with (cell_state+1) mod states
      is >= threshold.
    - neighborhood can be 4 (von Neumann) or 8 (Moore).
    - We assume wrapping (modulo) for edges.

    Returns the new grid after applying the rule.
    """
    new_grid = grid.copy()
    rows, cols = grid.shape

    # Precompute next_state array
    next_states = (grid + 1) % states

    # Determine neighbor offsets
    if neighborhood == 4:
        # von Neumann
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        # Moore
        neighbor_offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]

    for r in range(rows):
        for c in range(cols):
            ns = next_states[r, c]
            count_next_state = 0

            # Count neighbors matching ns
            for dr, dc in neighbor_offsets:
                rr = (r + dr) % rows
                cc = (c + dc) % cols
                if grid[rr, cc] == ns:
                    count_next_state += 1

            if count_next_state >= threshold:
                new_grid[r, c] = ns

    return new_grid

# -------------------------------------------------------------------
#                           Main App Class
# -------------------------------------------------------------------
class CyclicAutomatonApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Cyclic Cellular Automaton (Further Enhanced)")
        self.root.resizable(True, True)  # Allow window resizing

        # ------------- Simulation Variables -------------
        self.base_colors = [
            (0.0, 0.0, 0.0),   # black
            (1.0, 0.0, 0.0),   # red
            (1.0, 1.0, 0.0),   # yellow
            (1.0, 1.0, 1.0),   # white
        ]

        # By default, used for the current grid (width/height)
        self.grid_width = DEFAULT_GRID_WIDTH
        self.grid_height = DEFAULT_GRID_HEIGHT
        self.state_count = DEFAULT_NUM_COLORS

        self.speed = DEFAULT_SPEED  # Minimal delay => max speed
        self.running = False
        self.record = False

        # Create directory for frames if needed
        if not os.path.exists(IMG_DIRECTORY):
            os.makedirs(IMG_DIRECTORY)

        # ------------- UI Variables -------------
        # For spinboxes: number of colors, width, height
        self.num_colors_var = tk.IntVar(value=self.state_count)
        self.grid_width_var = tk.IntVar(value=self.grid_width)
        self.grid_height_var = tk.IntVar(value=self.grid_height)

        # Neighborhood & threshold
        self.neighborhood_var = tk.IntVar(value=8)  # 4 or 8
        self.threshold_var = tk.IntVar(value=1)

        # ------------- Create Initial Grid and Colormap -------------
        self.initial_grid = init_grid(self.grid_width, self.grid_height, self.state_count)
        self.grid = self.initial_grid.copy()

        self.cmap = create_gradient_cmap(self.base_colors, self.state_count)

        # ------------- Figure & Canvas -------------
        # Larger figure to be 3x bigger or simply large enough:
        self.fig, self.ax = plt.subplots(figsize=(15, 15))
        self.im = self.ax.imshow(
            self.grid,
            cmap=self.cmap,
            vmin=0,
            vmax=self.state_count - 1
        )
        self.ax.set_title("Cyclic Automaton")
        self.ax.axis("off")

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().grid(row=0, column=0, rowspan=12, sticky="nsew")

        # ------------- Control Panel -------------
        self.control_frame = ttk.Frame(self.root)
        self.control_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)

        # Play / Pause
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.start_simulation)
        self.play_button.grid(row=0, column=0, pady=5, sticky="ew")

        self.pause_button = ttk.Button(self.control_frame, text="Pause", command=self.pause_simulation)
        self.pause_button.grid(row=1, column=0, pady=5, sticky="ew")

        # Reset / Init
        self.reset_button = ttk.Button(self.control_frame, text="Reset", command=self.reset_grid)
        self.reset_button.grid(row=2, column=0, pady=5, sticky="ew")

        self.init_button = ttk.Button(self.control_frame, text="Init", command=self.init_grid_from_saved)
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

        # Neighborhood
        ttk.Label(self.control_frame, text="Neighborhood").grid(row=7, column=0, pady=5, sticky="ew")
        self.neighbor_combo = ttk.Combobox(self.control_frame, state="readonly")
        self.neighbor_combo["values"] = ["4 (von Neumann)", "8 (Moore)"]
        self.neighbor_combo.set("8 (Moore)")
        self.neighbor_combo.bind("<<ComboboxSelected>>", self.on_neighbor_change)
        self.neighbor_combo.grid(row=8, column=0, pady=5, sticky="ew")

        # Threshold
        ttk.Label(self.control_frame, text="Threshold").grid(row=9, column=0, pady=(5, 0), sticky="ew")
        self.threshold_spin = ttk.Spinbox(
            self.control_frame,
            from_=1, to=8,
            textvariable=self.threshold_var,
            width=5
        )
        self.threshold_spin.grid(row=10, column=0, pady=5, sticky="ew")

        # Number of Colors
        ttk.Label(self.control_frame, text="#Colors").grid(row=11, column=0, pady=(5, 0), sticky="ew")
        self.num_colors_spin = ttk.Spinbox(
            self.control_frame,
            from_=2, to=100,  # Arbitrary upper bound
            textvariable=self.num_colors_var,
            width=5
        )
        self.num_colors_spin.grid(row=12, column=0, pady=5, sticky="ew")

        # Grid Size Controls (Width / Height)
        ttk.Label(self.control_frame, text="Grid Width").grid(row=13, column=0, pady=(10, 0), sticky="ew")
        self.width_spin = ttk.Spinbox(
            self.control_frame,
            from_=10, to=2000,  # Arbitrary range
            textvariable=self.grid_width_var,
            width=5
        )
        self.width_spin.grid(row=14, column=0, pady=5, sticky="ew")

        ttk.Label(self.control_frame, text="Grid Height").grid(row=15, column=0, pady=(10, 0), sticky="ew")
        self.height_spin = ttk.Spinbox(
            self.control_frame,
            from_=10, to=2000,  # Arbitrary range
            textvariable=self.grid_height_var,
            width=5
        )
        self.height_spin.grid(row=16, column=0, pady=5, sticky="ew")

        # Configure resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Proper window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start the update loop (paused by default)
        self.update_loop()

    # ----------------------------------------------------------------
    #                       Simulation Control
    # ----------------------------------------------------------------
    def start_simulation(self):
        self.running = True

    def pause_simulation(self):
        self.running = False

    def reset_grid(self):
        """
        Create a new random grid with the user-specified width, height,
        and number of colors. Store that grid as 'initial_grid' and
        use it as current. Also recreate the colormap accordingly.
        """
        # Get new size parameters and number of colors
        self.grid_width = self.grid_width_var.get()
        self.grid_height = self.grid_height_var.get()
        self.state_count = self.num_colors_var.get()

        # Create new initial grid
        self.initial_grid = init_grid(self.grid_width, self.grid_height, self.state_count)
        self.grid = self.initial_grid.copy()

        # Recreate colormap with new state_count
        self.cmap = create_gradient_cmap(self.base_colors, self.state_count)

        # Update the image with new data
        self.im.set_data(self.grid)
        self.im.set_cmap(self.cmap)
        self.im.set_clim(vmin=0, vmax=self.state_count - 1)
        self.canvas.draw()

    def init_grid_from_saved(self):
        """
        Revert to the last stored initial grid.
        """
        self.grid = self.initial_grid.copy()
        self.im.set_data(self.grid)
        self.canvas.draw()

    def speed_up(self):
        """
        Decrease the update interval (go faster),
        minimal speed is 10 ms.
        """
        self.speed = max(10, self.speed - 50)

    def speed_down(self):
        """
        Increase the update interval (go slower).
        """
        self.speed += 50

    def toggle_record(self):
        self.record = self.record_var.get()

    # ----------------------------------------------------------------
    #                       Neighborhood & Threshold
    # ----------------------------------------------------------------
    def on_neighbor_change(self, event=None):
        """
        Update the internal neighborhood_var (4 or 8)
        based on the combobox selection.
        """
        choice = self.neighbor_combo.get()
        if choice.startswith("4"):
            self.neighborhood_var.set(4)
        else:
            self.neighborhood_var.set(8)

    # ----------------------------------------------------------------
    #                      Main Loop & Frame Saving
    # ----------------------------------------------------------------
    def update_loop(self):
        """
        Periodically update the grid if running,
        then schedule the next update.
        """
        if not self.root.winfo_exists():
            return  # Window closed, stop scheduling

        if self.running:
            # Update the grid
            neighborhood = self.neighborhood_var.get()
            threshold = self.threshold_var.get()

            self.grid = update_grid(
                self.grid,
                states=self.state_count,
                neighborhood=neighborhood,
                threshold=threshold
            )
            self.im.set_data(self.grid)
            self.canvas.draw()

            # Possibly record the frame
            if self.record:
                self.save_frame()

        # Schedule the next frame
        self.root.after(self.speed, self.update_loop)

    def save_frame(self):
        """
        Save the current frame as an image if recording is active.
        """
        global frame_counter
        frame_name = f"{IMG_DIRECTORY}/frame_{frame_counter:04d}.png"

        # Convert the grid to a PIL image via the colormap
        data_rgba = self.cmap(self.grid / max(1, (self.state_count - 1)))
        data_rgb = (data_rgba[:, :, :3] * 255).astype(np.uint8)

        img = Image.fromarray(data_rgb, mode='RGB')
        img.save(frame_name)
        frame_counter += 1

    def on_close(self):
        """
        Properly handle window closing.
        """
        self.running = False
        self.root.destroy()

# -------------------------------------------------------------------
#                                MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = CyclicAutomatonApp(root)
    root.mainloop()
