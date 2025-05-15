import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame
from typing import Optional, Tuple, Dict, Any

TIME_STEP_SIZE = 0.02
DRONE_MASS = 1.0
DRONE_INERTIA = 0.1
DRONE_RADIUS = 0.2
DRAG_COEFF = 0.05
OFFSET_LAT = 0.2
MAX_F_LONG = 1.0
MAX_F_LAT = 0.5
GRAVITY = 0.5
TOP_Y = 4.0
HOLE_WIDTH = 4.0 * DRONE_RADIUS
MAX_X = 2.5
MAX_V = 3.0
MAX_OMEGA = 3.0


class UnderwaterDrone:
    def __init__(
        self,
        seed=None,
        random_generator=None,
        m=DRONE_MASS,
        I=DRONE_INERTIA,
        Cd=DRAG_COEFF,
        radius=DRONE_RADIUS,
        offset_lateral=OFFSET_LAT,
        gravity=GRAVITY,
        max_F_long=MAX_F_LONG,
        max_F_lat=MAX_F_LAT,
        top_y=TOP_Y,
        hole_width=HOLE_WIDTH,
    ):
        # Initialize random state generator with seed
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        elif random_generator is None:
            self.rng = np.random.RandomState()
        else:
            self.rng = random_generator

        # Randomize initial state using the seeded generator
        self.x = self.rng.uniform(-MAX_X / 2, MAX_X / 2)
        self.y = self.rng.uniform(0, TOP_Y / 3)
        self.theta = self.rng.uniform(np.pi / 2 - np.pi / 20, np.pi / 2 + np.pi / 20)
        self.v_x = self.rng.uniform(-0.2, 0.2)
        self.v_y = self.rng.uniform(-0.2, 0.2)
        self.omega = self.rng.uniform(-0.2, 0.2)

        self.m = m
        self.I = I
        self.Cd = Cd
        self.radius = radius
        self.offset_lateral = offset_lateral
        self.gravity = gravity

        # Control bounds
        self.max_F_long = max_F_long  # max forward/backward thrust
        self.max_F_lat = max_F_lat  # max lateral thrust

        # To detect getting into air
        self.top_y = top_y
        self.hole_width = hole_width

        # We'll keep a "frozen" flag
        self.frozen = False

        # Prepare local geometry for the "nose" polygon
        self.nose_len = 0.3
        self.nose_half_w = 0.2
        # local coords: tip at (nose_len, 0), base corners around (0, +/- nose_half_w)
        self.nose_local_coords = np.array(
            [[self.nose_len, 0.0], [0.0, self.nose_half_w], [0.0, -self.nose_half_w]]
        )

    def step(self, action, dt=TIME_STEP_SIZE):
        """
        action = (F_long, F_lat)
          F_long: thrust in the drone's longitudinal direction (body x-axis).
          F_lat:  thrust in the drone's lateral direction (body y-axis).
        dt: time step for Euler integration.
        """
        if self.frozen:
            return

        if self._in_hole():
            self._freeze()
            return

        F_long, F_lat = action
        F_long = np.clip(F_long, -self.max_F_long, self.max_F_long)
        F_lat = np.clip(F_lat, -self.max_F_lat, self.max_F_lat)

        c = np.cos(self.theta)
        s = np.sin(self.theta)
        R = np.array([[c, -s], [s, c]])

        thrust_body = np.array([F_long, F_lat])
        thrust_inertial = R @ thrust_body

        v = np.array([self.v_x, self.v_y])
        speed = np.linalg.norm(v)
        if speed > 1e-6:
            drag_dir = -v / speed
        else:
            drag_dir = np.array([0.0, 0.0])
        F_drag = self.Cd * speed**2 * drag_dir

        F_net = thrust_inertial + F_drag + np.array([0.0, -self.gravity])

        tau = self.offset_lateral * F_lat

        a_x = F_net[0] / self.m
        a_y = F_net[1] / self.m
        alpha = tau / self.I

        self.v_x += a_x * dt
        self.v_y += a_y * dt
        self.omega += alpha * dt

        self.x += self.v_x * dt
        self.y += self.v_y * dt
        self.theta += self.omega * dt

        self.x = np.clip(self.x, -MAX_X - 0.01, MAX_X + 0.01)
        self.y = np.clip(self.y, 0.0 - 0.01, TOP_Y + 0.01)

        if self._in_hole():
            self._freeze()
            return

        if self.y < 0.0:
            self.v_y = max(self.v_y, 0)
        if self.y > TOP_Y:
            self.v_y = min(self.v_y, 0)
        if self.x < -MAX_X:
            self.v_x = max(self.v_x, 0)
        if self.x > MAX_X:
            self.v_x = min(self.v_x, 0)

    def _in_hole(self):
        """
        Return True if the drone's center is at/above top_y
        AND within the horizontal hole region, i.e. x in [-hole_half, hole_half].
        """
        hole_half = self.hole_width / 2.0
        if self.y >= self.top_y:
            if -hole_half <= self.x <= hole_half:
                return True
        return False

    def _near_borders(self):
        """
        Return True if the drone's center is near the borders.
        """
        return (
            np.abs(self.x) > MAX_X - 0.01
            or self.y < 0.0 + 0.01
            or (
                np.abs(self.x) > self.hole_width / 2.0 + 0.01 and self.y >= TOP_Y - 0.01
            )
        )

    def _freeze(self):
        """Freeze the drone's motion. Zero velocities, 'frozen'=True."""
        self.v_x = 0.0
        self.v_y = 0.0
        self.omega = 0.0
        self.frozen = True

    def state(self):
        """Return (x, y, theta, v_x, v_y, omega)."""
        return np.array(
            [self.x, self.y, self.theta, self.v_x, self.v_y, self.omega],
            dtype=np.float32,
        )


class UnderwaterDroneEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": int(1.0 / TIME_STEP_SIZE),
    }

    def __init__(self, render_mode: Optional[str] = None, seed: Optional[int] = None):
        # Define observation space
        # State is (x, y, theta, v_x, v_y, omega)
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.inf, -np.inf, -1, -1, -np.inf, -np.inf, -np.inf], dtype=np.float32
            ),
            high=np.array(
                [np.inf, np.inf, 1, 1, np.inf, np.inf, np.inf], dtype=np.float32
            ),
            dtype=np.float32,
        )
        self.rng = np.random.RandomState(seed)

        self.semimajor_axis = 0.9
        self.semiminor_axis = 0.6

        # Define action space
        # Action is (F_long, F_lat)
        self.action_space = spaces.Box(
            low=np.array([-MAX_F_LONG, -MAX_F_LAT], dtype=np.float32),
            high=np.array([MAX_F_LONG, MAX_F_LAT], dtype=np.float32),
            dtype=np.float32,
        )

        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.drone = None

        # Display settings
        self.screen_width = 800
        self.screen_height = 600
        self.scale_factor = 100  # Scale from physics units to pixels

        # Origin in screen coordinates (bottom-left in physics)
        self.origin_x = self.screen_width // 2
        self.origin_y = self.screen_height - 100

        # Trajectory tracking
        self.trajectory = []

        # Heatmap surface
        self.heatmap_surface = None

        # Counters
        self.n_near_borders = 0
        self.n_in_spot = 0
        self.n_resets = 0
        self.avoidance_score = np.inf
        # Reset the environment
        self.reset()

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.n_resets += 1

        if seed is not None:
            self.drone = UnderwaterDrone(seed=seed)
        elif self.rng is not None:
            self.drone = UnderwaterDrone(random_generator=self.rng)
        else:
            self.drone = UnderwaterDrone()

        # Clear trajectory history
        self.trajectory = []

        # Setup for rendering
        self._setup_rendering()
        self.n_near_borders = 0
        self.n_in_spot = 0
        self.avoidance_score = -np.inf
        return self._get_obs(), self._get_info()

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Execute drone physics
        self.drone.step(action, dt=TIME_STEP_SIZE)

        # Track trajectory
        self.trajectory.append((self.drone.x, self.drone.y))

        # Calculate reward
        reward = self._calculate_reward()

        terminated = False
        truncated = False

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _is_in_spot(self):
        return (
            self.drone.x ** 2 / self.semimajor_axis**2
            + (self.drone.y - TOP_Y / 2) ** 2 / self.semiminor_axis**2
            <= 1.0
        )

    def _calculate_reward(self) -> float:
        # Simple reward: -1 if frozen, otherwise 0.01 for each step plus height bonus
        return (
            -0.25 * ((self.drone.y - TOP_Y) ** 2)
            - 0.25 * (self.drone.x - 0.0) ** 2
            - 0.05 * self.drone.v_x**2
            - 0.05 * self.drone.v_y**2
            - 0.01 * self.drone.omega**2
            - 5 * (1 if self._is_in_spot() else 0)
        )

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.drone.x,
                self.drone.y,
                np.cos(self.drone.theta),
                np.sin(self.drone.theta),
                self.drone.v_x,
                self.drone.v_y,
                self.drone.omega,
            ],
            dtype=np.float32,
        )

    def _get_info(self) -> Dict[str, Any]:
        if self.drone._near_borders():
            self.n_near_borders += 1
        if self._is_in_spot():
            self.n_in_spot += 1

        current_avoidance_score = np.clip(
            self.drone.x ** 2 / self.semimajor_axis**2
            + (self.drone.y - TOP_Y / 2) ** 2 / self.semiminor_axis**2,
            0.0,
            1.0,
        )
        self.avoidance_score = min(self.avoidance_score, current_avoidance_score)

        return {
            "x": self.drone.x,
            "y": self.drone.y,
            "is_frozen": self.drone.frozen,
            "is_in_hole": self.drone._in_hole(),
            "is_near_borders": self.drone._near_borders(),
            "n_near_borders": self.n_near_borders,
            "is_in_spot": self._is_in_spot(),
            "n_in_spot": self.n_in_spot,
            "current_avoidance_score": current_avoidance_score,
            "avoidance_score": np.copy(self.avoidance_score),
        }

    def _setup_rendering(self):
        if self.render_mode is None:
            return

        if self.screen is None and self.render_mode in ["human", "rgb_array"]:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Underwater Drone Simulator")
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))

            self.clock = pygame.time.Clock()

            # Create heatmap surface
            self._create_heatmap()

    def _create_heatmap(self):
        """Create a surface with the ellipse visualization based on _is_in_spot method"""
        if self.screen is None:
            return

        # Create a transparent surface for the heatmap
        self.heatmap_surface = pygame.Surface(
            (self.screen_width, self.screen_height), pygame.SRCALPHA
        )

        # Ellipse parameters
        ellipse_center_x = 0.0
        ellipse_center_y = TOP_Y / 2
        ellipse_a = self.semimajor_axis  # semi-major axis
        ellipse_b = self.semiminor_axis  # semi-minor axis

        # Convert ellipse parameters to screen coordinates
        ellipse_center_px = self.to_pixels_x(ellipse_center_x)
        ellipse_center_py = self.to_pixels_y(ellipse_center_y)
        ellipse_width = int(2 * ellipse_a * self.scale_factor)
        ellipse_height = int(2 * ellipse_b * self.scale_factor)

        # Calculate the rectangle that bounds the ellipse
        ellipse_rect = pygame.Rect(
            ellipse_center_px - ellipse_width // 2,
            ellipse_center_py - ellipse_height // 2,
            ellipse_width,
            ellipse_height,
        )

        # Draw the ellipse with a semi-transparent fill
        pygame.draw.ellipse(
            self.heatmap_surface, (255, 0, 0, 80), ellipse_rect  # Red with alpha
        )

        # Draw the ellipse outline
        pygame.draw.ellipse(
            self.heatmap_surface,
            (255, 0, 0, 160),  # Red with higher alpha for the outline
            ellipse_rect,
            width=2,
        )

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode is None:
            return None

        if self.screen is None:
            self._setup_rendering()

        # Fill the screen with a blue background (water)
        self.screen.fill((25, 25, 150))

        # Draw the top surface and hole
        pygame.draw.rect(
            self.screen,
            (135, 206, 235),  # Sky blue
            pygame.Rect(
                0, 0, self.screen_width, self.screen_height - self.to_pixels_y(TOP_Y)
            ),
        )

        # Draw the hole
        hole_half_width = self.drone.hole_width / 2.0
        hole_left = self.to_pixels_x(-hole_half_width)
        hole_right = self.to_pixels_x(hole_half_width)
        hole_top = self.to_pixels_y(TOP_Y)
        pygame.draw.rect(
            self.screen,
            (25, 25, 150),  # Water color
            pygame.Rect(hole_left, 0, hole_right - hole_left, hole_top),
        )

        # Draw the heatmap
        if self.heatmap_surface is not None:
            self.screen.blit(self.heatmap_surface, (0, 0))

        # Draw environment boundaries with thick black lines
        # Left boundary
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black
            (self.to_pixels_x(-MAX_X), self.to_pixels_y(0)),
            (self.to_pixels_x(-MAX_X), self.to_pixels_y(TOP_Y)),
            4,  # Line thickness
        )
        # Right boundary
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black
            (self.to_pixels_x(MAX_X), self.to_pixels_y(0)),
            (self.to_pixels_x(MAX_X), self.to_pixels_y(TOP_Y)),
            4,  # Line thickness
        )
        # Bottom boundary
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black
            (self.to_pixels_x(-MAX_X), self.to_pixels_y(0)),
            (self.to_pixels_x(MAX_X), self.to_pixels_y(0)),
            4,  # Line thickness
        )
        # Top boundary (except the hole)
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black
            (self.to_pixels_x(-MAX_X), self.to_pixels_y(TOP_Y)),
            (self.to_pixels_x(-hole_half_width), self.to_pixels_y(TOP_Y)),
            4,  # Line thickness
        )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),  # Black
            (self.to_pixels_x(hole_half_width), self.to_pixels_y(TOP_Y)),
            (self.to_pixels_x(MAX_X), self.to_pixels_y(TOP_Y)),
            4,  # Line thickness
        )

        # Draw trajectory as blue dotted line
        if len(self.trajectory) > 1:
            points = [
                (self.to_pixels_x(x), self.to_pixels_y(y)) for x, y in self.trajectory
            ]
            pygame.draw.lines(
                self.screen,
                (0, 0, 255),  # Blue
                False,  # Not closed
                points,
                2,  # Line width
            )

            # Add dotted effect
            for i, point in enumerate(points):
                if i % 3 == 0:  # Adjust spacing of dots
                    pygame.draw.circle(
                        self.screen, (255, 255, 255), point, 1  # White  # Dot radius
                    )

        # Draw the drone body
        pygame.draw.circle(
            self.screen,
            (0, 100, 0),  # Dark green
            (self.to_pixels_x(self.drone.x), self.to_pixels_y(self.drone.y)),
            int(self.drone.radius * self.scale_factor),
        )

        # Draw the drone nose
        c = np.cos(self.drone.theta)
        s = np.sin(self.drone.theta)
        R = np.array([[c, -s], [s, c]])

        tri_world = (R @ self.drone.nose_local_coords.T).T + np.array(
            [self.drone.x, self.drone.y]
        )
        tri_pixels = [
            (self.to_pixels_x(point[0]), self.to_pixels_y(point[1]))
            for point in tri_world
        ]

        pygame.draw.polygon(self.screen, (139, 0, 0), tri_pixels)  # Dark red

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )

    def to_pixels_x(self, x: float) -> int:
        return int(self.origin_x + x * self.scale_factor)

    def to_pixels_y(self, y: float) -> int:
        return int(self.origin_y - y * self.scale_factor)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
