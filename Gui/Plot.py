from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QMainWindow
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
import numpy as np

# --- Cylinder mesh helpers ---

_AXIS_X = np.array([1.0, 0.0, 0.0])
_AXIS_Y = np.array([0.0, 1.0, 0.0])

# Cache of theta samples (unit-circle angle grids) keyed by n_theta, so we
# don't call np.linspace repeatedly for every link/joint on every frame.
_THETA_CACHE = {}


def _theta_for(n_theta):
    theta = _THETA_CACHE.get(n_theta)
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, n_theta)
        _THETA_CACHE[n_theta] = theta
    return theta


def cylinder_between(p0, p1, radius, n_theta=16, caps=True):
    """
    Build a cylinder surface mesh whose central axis runs from p0 to p1.

    Returns X, Y, Z arrays suitable for ax.plot_surface(X, Y, Z, ...).
    """
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    v = p1 - p0
    height = np.linalg.norm(v)
    if height < 1e-9:
        return None

    v = v / height

    # Any vector not parallel to v, used to build an orthonormal frame.
    not_v = _AXIS_Y if abs(np.dot(v, _AXIS_X)) > 0.9 else _AXIS_X

    n1 = np.cross(v, not_v)
    n1 /= np.linalg.norm(n1)
    n2 = np.cross(v, n1)

    theta = _theta_for(n_theta)
    t = np.array([0.0, height])

    # circle offsets in the plane perpendicular to v: shape (n_theta, 3)
    circle = radius * (np.outer(np.sin(theta), n1) +
                       np.outer(np.cos(theta), n2))
    # base points along the axis: shape (2, 3)
    base = p0 + np.outer(t, v)
    # broadcast to shape (2, n_theta, 3)
    points = base[:, None, :] + circle[None, :, :]

    return points[..., 0], points[..., 1], points[..., 2]


def cylinder_at(center, axis, radius, length, n_theta=20):
    """
    Build a cylinder centered at `center`, aligned with `axis`
    (does not need to be unit length), with total `length`.
    Use this for joint "pucks" whose orientation = joint rotation axis.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    p0 = np.asarray(center) - axis * (length / 2.0)
    p1 = np.asarray(center) + axis * (length / 2.0)
    return cylinder_between(p0, p1, radius, n_theta=n_theta)


# --- Plot Window ---


class PlotWindow(QMainWindow):
    def __init__(self, robot):
        super().__init__()
        self.robot = robot
        self.setWindowTitle("3D Robot Visualization")
        self.figure = Figure(figsize=(8, 8), dpi=100)
        self.figure.subplots_adjust(
            top=0.958,
            bottom=0.015,
            left=0.013,
            right=0.987,
            hspace=0.2,
            wspace=0.2
        )

        self.canvas = FigureCanvas(self.figure)

        # --- Add Reset Button ---
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        reset_button = QPushButton("Reset View")
        reset_button.clicked.connect(self.reset_view)
        self.toolbar.addWidget(reset_button)

        # Layout to include the toolbar
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.ax = self.figure.add_subplot(111, projection="3d")

        # Store default view angles (elevation, azimuth)
        self.default_view = (30, 45)
        self.ax.view_init(elev=self.default_view[0], azim=self.default_view[1])

        # Style and setup...
        self.ax.set_facecolor('white')
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # --- Cylinder rendering state (replaces the old self.line) ---
        # Kept in a single list (rather than two) so the per-frame teardown
        # is one loop instead of two.
        self.surfaces = []

        # Scale these relative to robot.reach so they look right regardless
        # of whether your robot is a tabletop arm (reach ~0.5m) or a big
        # industrial one (reach ~2m+).
        self.link_radius = 0.025 * self.robot.reach
        self.joint_radius = 0.045 * self.robot.reach
        self.joint_length = 0.09 * self.robot.reach

        # Pre-warm the theta cache for the resolutions we'll actually use.
        _theta_for(16)
        _theta_for(20)

        self.setup_axes()

    def reset_view(self):
        """Resets the camera to the default orientation."""
        self.ax.view_init(elev=self.default_view[0], azim=self.default_view[1])
        self.canvas.draw_idle()

    def setup_axes(self):
        r = self.robot.reach - 0.25
        self.ax.set_xlim(-r, r)
        self.ax.set_ylim(-r, r)
        self.ax.set_zlim(-0.1, r)

        # Add labels
        self.ax.set_title(
            f"{self.robot.name} | {self.robot.n}-DOF Robot",
            fontsize=20,
            fontweight='bold',
            pad=5,  # Optional: adds space between title and plot
            y=1.05
        )
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')

        # Add a ground plane (simple grid)
        x = np.linspace(-r, r, 5)
        y = np.linspace(-r, r, 5)
        X, Y = np.meshgrid(x, y)
        Z = np.zeros_like(X)
        self.ax.plot_surface(X, Y, Z, color='gray', alpha=0.1)

    def update_view(self, positions, axes):
        """
        positions : (N+1, 3) array — joint/EE positions from fkine_all,
                    base -> tip.
        axes      : (N+1, 3) array — world-frame rotation axis (z-axis of
                    each transform's rotation matrix) at each position.
        """
        # Remove last frame's surfaces before drawing the new pose.
        for s in self.surfaces:
            s.remove()
        self.surfaces.clear()

        # Links: a tube between every consecutive pair of frames.
        for p0, p1 in zip(positions[:-1], positions[1:]):
            mesh = cylinder_between(p0, p1, self.link_radius, n_theta=16)
            if mesh is None:
                continue
            X, Y, Z = mesh
            surf = self.ax.plot_surface(
                X, Y, Z, color='#1f77b4', edgecolor='none',
                shade=True, antialiased=True
            )
            self.surfaces.append(surf)

        # Joints: a fatter puck at each frame, oriented along its z-axis.
        for center, axis in zip(positions, axes):
            mesh = cylinder_at(
                center, axis, self.joint_radius, self.joint_length, n_theta=20)
            if mesh is None:
                continue
            X, Y, Z = mesh
            surf = self.ax.plot_surface(
                X, Y, Z, color='#d62728', edgecolor='none',
                shade=True, antialiased=True
            )
            self.surfaces.append(surf)

        self.canvas.draw_idle()
