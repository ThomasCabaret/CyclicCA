import sys
import time
import glfw
import moderngl
import numpy as np
import pyrr

"""
Fixed GPU Particle Simulation
- Aligns the Particle struct to 32 bytes (8 floats).
- Still only does wall collisions (no particle-particle collisions).
- Gradient color (blue?red?yellow) based on speed.
- Controls:
    Space = pause/play
    R     = reset
"""

# ---------------- Configuration ---------------
NUM_PARTICLES   = 1000
DOMAIN_SIZE     = 1.0
PARTICLE_RADIUS = 0.01
MAX_VELOCITY    = 0.3
WINDOW_WIDTH    = 800
WINDOW_HEIGHT   = 800

COMPUTE_SHADER_SRC = """
#version 430
layout(local_size_x = 256) in;

// We force each Particle to be 8 floats = 32 bytes (vec2 pos + vec2 vel + float speed + 3 floats pad).
struct Particle {
    vec2 pos;    // 2 floats
    vec2 vel;    // 2 floats
    float speed; // 1 float
    float pad0;
    float pad1;
    float pad2;  // total = 2+2+1+3 = 8 floats => 32 bytes
};

layout(std430, binding = 0) buffer Particles {
    Particle particles[];
};

uniform float dt;
uniform float domain;
uniform float radius;
uniform float elasticity;

void main() {
    uint i = gl_GlobalInvocationID.x;
    Particle p = particles[i];

    // 1) Update position
    p.pos += p.vel * dt;

    // 2) Wall collisions
    if (p.pos.x < radius) {
        p.pos.x = radius;
        p.vel.x = abs(p.vel.x) * elasticity;
    } else if (p.pos.x > domain - radius) {
        p.pos.x = domain - radius;
        p.vel.x = -abs(p.vel.x) * elasticity;
    }

    if (p.pos.y < radius) {
        p.pos.y = radius;
        p.vel.y = abs(p.vel.y) * elasticity;
    } else if (p.pos.y > domain - radius) {
        p.pos.y = domain - radius;
        p.vel.y = -abs(p.vel.y) * elasticity;
    }

    // 3) Compute speed
    p.speed = length(p.vel);

    // Store updated
    particles[i] = p;
}
""";

VERTEX_SHADER_SRC = """
#version 430

// We'll read (pos.x, pos.y, speed, pad0, pad1, pad2) from the VBO, ignoring velocity.
in vec2 in_pos;    // 2 floats
in float in_speed; // 1 float

uniform mat4  u_proj;
uniform float max_speed;
uniform float u_pointsize;

out float particle_speed;

void main() {
    gl_Position = u_proj * vec4(in_pos, 0.0, 1.0);
    gl_PointSize = u_pointsize;

    particle_speed = in_speed / max_speed; // normalized speed
}
""";

FRAGMENT_SHADER_SRC = """
#version 430

in float particle_speed;
out vec4 f_color;

void main() {
    vec2 pc = gl_PointCoord - vec2(0.5);
    float dist = length(pc);
    if (dist > 0.5) {
        discard;
    }

    // Blue -> Red -> Yellow
    if (particle_speed < 0.5) {
        f_color = mix(vec4(0.0, 0.0, 1.0, 1.0), vec4(1.0, 0.0, 0.0, 1.0), particle_speed * 2.0);
    } else {
        f_color = mix(vec4(1.0, 0.0, 0.0, 1.0), vec4(1.0, 1.0, 0.0, 1.0), (particle_speed - 0.5) * 2.0);
    }
}
""";

class GPUParticleSimulation:
    def __init__(self, width, height, title="GPU Particle Simulation"):
        if not glfw.init():
            raise RuntimeError("Failed to init GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)

        self.window = glfw.create_window(width, height, title, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self.window)
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND)
        self.ctx.enable(moderngl.PROGRAM_POINT_SIZE)
        self.ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)

        self.width  = width
        self.height = height
        self.running = True

        # ----- Shaders -----
        self.compute_prog = self.ctx.compute_shader(COMPUTE_SHADER_SRC)
        self.render_prog  = self.ctx.program(
            vertex_shader=VERTEX_SHADER_SRC,
            fragment_shader=FRAGMENT_SHADER_SRC
        )

        # Uniforms
        self.dt_uniform        = self.compute_prog["dt"]
        self.domain_uniform    = self.compute_prog["domain"]
        self.radius_uniform    = self.compute_prog["radius"]
        self.elasticity_uniform= self.compute_prog["elasticity"]

        self.u_proj       = self.render_prog["u_proj"]
        self.u_pointsize  = self.render_prog["u_pointsize"]
        self.u_max_speed  = self.render_prog["max_speed"]

        self.num_particles = NUM_PARTICLES

        # ----- Create CPU array, sized 8 floats = 32 bytes each -----
        # (pos.x, pos.y, vel.x, vel.y, speed, pad0, pad1, pad2)
        dtype_particle = np.dtype([
            ("posx",  np.float32),
            ("posy",  np.float32),
            ("velx",  np.float32),
            ("vely",  np.float32),
            ("speed", np.float32),
            ("pad0",  np.float32),
            ("pad1",  np.float32),
            ("pad2",  np.float32),
        ])

        part_data = np.zeros(self.num_particles, dtype=dtype_particle)

        # Random init
        part_data["posx"] = np.random.uniform(
            PARTICLE_RADIUS, DOMAIN_SIZE - PARTICLE_RADIUS, self.num_particles
        ).astype(np.float32)
        part_data["posy"] = np.random.uniform(
            PARTICLE_RADIUS, DOMAIN_SIZE - PARTICLE_RADIUS, self.num_particles
        ).astype(np.float32)
        part_data["velx"] = np.random.uniform(
            -MAX_VELOCITY, MAX_VELOCITY, self.num_particles
        ).astype(np.float32)
        part_data["vely"] = np.random.uniform(
            -MAX_VELOCITY, MAX_VELOCITY, self.num_particles
        ).astype(np.float32)
        part_data["speed"] = 0.0

        self.ssbo = self.ctx.buffer(part_data.tobytes())
        self.ssbo.bind_to_storage_buffer(0)

        # Create a VBO for rendering (pos.x, pos.y, speed => total 3 floats)
        # We'll skip vel + pads
        self.vbo = self.ctx.buffer(reserve=self.num_particles * 3 * 4)
        self.vao = self.ctx.vertex_array(
            self.render_prog,
            [
                (self.vbo, "2f 1f", "in_pos", "in_speed"),
            ],
        )

        # Projection
        proj = pyrr.matrix44.create_orthogonal_projection(
            0.0, DOMAIN_SIZE, 0.0, DOMAIN_SIZE, -1.0, 1.0
        )
        self.u_proj.write(proj.astype("f4").tobytes())

        self.u_pointsize.value = 2.0 * PARTICLE_RADIUS * self.width
        self.u_max_speed.value = MAX_VELOCITY
        self.domain_uniform.value    = DOMAIN_SIZE
        self.radius_uniform.value    = PARTICLE_RADIUS
        self.elasticity_uniform.value= 1.0

        self.last_time = time.time()

        glfw.set_key_callback(self.window, self.on_key)

    def on_key(self, window, key, scancode, action, mods):
        if action == glfw.PRESS:
            if key == glfw.KEY_SPACE:
                self.running = not self.running
            elif key == glfw.KEY_R:
                self.reset_particles()

    def reset_particles(self):
        # Reinitialize random data
        dtype_particle = np.dtype([
            ("posx",  np.float32),
            ("posy",  np.float32),
            ("velx",  np.float32),
            ("vely",  np.float32),
            ("speed", np.float32),
            ("pad0",  np.float32),
            ("pad1",  np.float32),
            ("pad2",  np.float32),
        ])
        part_data = np.zeros(self.num_particles, dtype=dtype_particle)

        part_data["posx"] = np.random.uniform(
            PARTICLE_RADIUS, DOMAIN_SIZE - PARTICLE_RADIUS, self.num_particles
        ).astype(np.float32)
        part_data["posy"] = np.random.uniform(
            PARTICLE_RADIUS, DOMAIN_SIZE - PARTICLE_RADIUS, self.num_particles
        ).astype(np.float32)
        part_data["velx"] = np.random.uniform(
            -MAX_VELOCITY, MAX_VELOCITY, self.num_particles
        ).astype(np.float32)
        part_data["vely"] = np.random.uniform(
            -MAX_VELOCITY, MAX_VELOCITY, self.num_particles
        ).astype(np.float32)
        part_data["speed"] = 0.0
        self.ssbo.write(part_data.tobytes())

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            if self.running:
                self.update_sim()
            self.render()
            glfw.swap_buffers(self.window)
        glfw.terminate()

    def update_sim(self):
        now = time.time()
        dt = min(now - self.last_time, 0.02)
        self.last_time = now

        self.compute_prog["dt"].value = dt

        group_size = 256
        num_groups = (self.num_particles + group_size - 1) // group_size
        self.compute_prog.run(num_groups, 1, 1)

    def render(self):
        # Copy only (pos.x, pos.y, speed)
        #   offsets: pos.x=0, pos.y=4, vel.x=8, vel.y=12, speed=16 => pad0=20 => ...
        # But we have 8 floats (32 bytes) per particle total. We'll read entire buffer,
        # then slice out columns [posx, posy, speed].
        raw_data = self.ssbo.read()
        arr = np.frombuffer(raw_data, dtype=np.float32).reshape(self.num_particles, 8)
        # columns: 0=posx, 1=posy, 2=velx, 3=vely, 4=speed, 5=pad0, ...
        copy_data = arr[:, [0, 1, 4]]  # shape (n,3)
        self.vbo.write(copy_data.tobytes())

        self.ctx.clear(0, 0, 0, 1)
        self.vao.render(moderngl.POINTS, vertices=self.num_particles)


def main():
    sim = GPUParticleSimulation(WINDOW_WIDTH, WINDOW_HEIGHT)
    sim.run()

if __name__ == "__main__":
    main()
