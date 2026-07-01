from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QMessageBox, QApplication, QSlider, QGroupBox,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from spatialmath import SE3
import numpy as np
import sys

from Gui.plot import PlotWindow

# --- Control Panel Window ---


class ControlPanel(QWidget):
    def __init__(self, robot, update_callback, move_callback, reset_callback, close_program):
        super().__init__()
        self.robot = robot
        self.resize(400, 715)
        self.setWindowTitle("Robot Controls")
        layout = QVBoxLayout(self)

        coord_box = QGroupBox("Joint & End-Effector Positions")
        coord_layout = QVBoxLayout()

        # Create table: N joints + 1 end-effector
        self.coord_table = QTableWidget(self.robot.n + 1, 3)
        self.coord_table.setHorizontalHeaderLabels(["X (m)", "Y (m)", "Z (m)"])

        row_labels = [
            f"Joint {i+1}" for i in range(self.robot.n)] + ["End Effector"]
        self.coord_table.setVerticalHeaderLabels(row_labels)
        self.coord_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch)

        # Pre-create table items once and reuse them via setText() on every
        # update instead of allocating a new QTableWidgetItem every frame.
        self._table_items = []
        for i in range(self.robot.n + 1):
            row_items = []
            for j in range(3):
                item = QTableWidgetItem("0.000")
                self.coord_table.setItem(i, j, item)
                row_items.append(item)
            self._table_items.append(row_items)

        coord_layout.addWidget(self.coord_table)
        coord_box.setLayout(coord_layout)
        layout.addWidget(coord_box)

        # Joint Sliders
        joint_box = QGroupBox("Joint Angles")
        joint_layout = QVBoxLayout()
        self.sliders = []
        self.val_labels = []
        for i in range(self.robot.n):
            row = QHBoxLayout()
            label = QLabel(f"Joint {i+1}")
            slider = QSlider(Qt.Orientation.Horizontal)
            qlim = np.degrees(self.robot.links[i].qlim)
            slider.setMinimum(int(qlim[0]))
            slider.setMaximum(int(qlim[1]))

            slider.setValue(0)
            val_label = QLabel("0°")
            slider.valueChanged.connect(
                lambda v, idx=i, l=val_label: update_callback(idx, v, l))
            self.sliders.append(slider)
            self.val_labels.append(val_label)
            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(val_label)
            joint_layout.addLayout(row)
        joint_box.setLayout(joint_layout)
        layout.addWidget(joint_box)

        # Target Inputs
        target_box = QGroupBox("Target Position (m)")
        target_layout = QHBoxLayout()
        self.inputs = {}
        for axis in ["X", "Y", "Z"]:
            box = QVBoxLayout()
            edit = QLineEdit("0.0")
            box.addWidget(QLabel(axis))
            box.addWidget(edit)
            self.inputs[axis] = edit
            target_layout.addLayout(box)
        target_box.setLayout(target_layout)
        layout.addWidget(target_box)

        # Buttons
        btn_layout = QHBoxLayout()
        move_btn = QPushButton("Move to XYZ")
        reset_btn = QPushButton("Reset")
        quit = QPushButton("Quit")

        move_btn.clicked.connect(lambda: move_callback(self.inputs))
        reset_btn.clicked.connect(reset_callback)
        quit.clicked.connect(close_program)

        btn_layout.addWidget(move_btn)
        btn_layout.addWidget(reset_btn)
        btn_layout.addWidget(quit)

        layout.addLayout(btn_layout)

    def update_coordinate_table(self, xyz_data):
        """xyz_data is an (N+1, 3) array."""
        for i, pos in enumerate(xyz_data):
            row_items = self._table_items[i]
            for j in range(3):
                row_items[j].setText(f"{pos[j]:.3f}")

    def set_joint_display(self, idx, degrees):
        """Update a slider + its label without emitting valueChanged,
        so callers can batch-update many joints and only trigger one
        downstream plot refresh."""
        slider = self.sliders[idx]
        slider.blockSignals(True)
        slider.setValue(int(degrees))
        slider.blockSignals(False)
        self.val_labels[idx].setText(f"{int(degrees)}°")

# --- Main Orchestrator ---


class RobotControllerGUI:
    def __init__(self, robot):
        self.robot = robot
        self.plot_win = PlotWindow(robot)
        self.control_win = ControlPanel(
            robot, self.slider_move, self.move_robot, self.reset_robot, self.close_program)

        self.plot_win.show()
        self.control_win.show()
        self.update_plot()

    def slider_move(self, joint, value, label):
        label.setText(f"{value}°")
        q = self.robot.q.copy()
        q[joint] = np.radians(value)
        self.robot.q = q
        self.update_plot()

    def move_robot(self, inputs):
        try:
            target = np.array([float(inputs["X"].text()), float(
                inputs["Y"].text()), float(inputs["Z"].text())])

            q_min = self.robot.qlim[0, :]
            q_max = self.robot.qlim[1, :]

            def wrap_to_limits(q):
                """Wrap each joint angle by 2*pi to try to land inside its limits."""
                q_wrapped = q.copy()
                for i in range(len(q_wrapped)):
                    # try shifting by -2pi, 0, +2pi and pick whichever is in range
                    candidates = [q_wrapped[i] + 2 *
                                  np.pi * k for k in (-1, 0, 1)]
                    in_range = [
                        c for c in candidates if q_min[i] <= c <= q_max[i]]
                    if in_range:
                        # pick the candidate closest to the original value
                        q_wrapped[i] = min(
                            in_range, key=lambda c: abs(c - q[i]))
                return q_wrapped

            def check_limits(q):
                out_of_bounds = []
                for i in range(len(q)):
                    if q[i] < q_min[i] or q[i] > q_max[i]:
                        out_of_bounds.append(
                            f"Joint {i+1} ({np.degrees(q[i]):.1f}° exceeds "
                            f"{np.degrees(q_min[i]):.1f}°-{np.degrees(q_max[i]):.1f}°)")
                return out_of_bounds

            # Try current config first, then a few random restarts if needed
            seeds = [self.robot.q] + [
                np.random.uniform(q_min, q_max) for _ in range(20)
            ]

            best_q = None
            best_errors = None

            for q0 in seeds:
                sol = self.robot.ikine_LM(
                    SE3(target), q0=q0, mask=[1, 1, 1, 0, 0, 0])

                if not sol.success:
                    continue

                q_fixed = wrap_to_limits(sol.q)
                out_of_bounds = check_limits(q_fixed)

                if not out_of_bounds:
                    best_q = q_fixed
                    break
                elif best_errors is None:
                    # keep the first failing attempt around for error reporting
                    best_errors = out_of_bounds

            if best_q is not None:
                self.animate_to(best_q)
            elif best_errors is not None:
                error_msg = "Solution rejected due to joint limits:\n" + \
                    "\n".join(best_errors)
                QMessageBox.warning(None, "Joint Limit Error", error_msg)
            else:
                QMessageBox.warning(None, "IK", "Unreachable")

        except Exception as e:
            QMessageBox.warning(None, "Error", str(e))

    def animate_to(self, q_target, duration_ms=800, steps=40):
        """Smoothly interpolate joint angles from current pose to q_target."""
        # Stop any animation already in progress so clicks don't stack
        if hasattr(self, "_anim_timer") and self._anim_timer.isActive():
            self._anim_timer.stop()

        self._anim_q_start = np.array(self.robot.q, dtype=float)
        self._anim_q_target = np.array(q_target, dtype=float)
        self._anim_step = 0
        self._anim_steps = steps

        self._anim_timer = QTimer()
        self._anim_timer.timeout.connect(self._animate_step)
        self._anim_timer.start(max(1, duration_ms // steps))

    def _animate_step(self):
        self._anim_step += 1
        t = self._anim_step / self._anim_steps

        # Ease-in-out so the move isn't linear/robotic-looking
        t_eased = 0.5 - 0.5 * np.cos(np.pi * t)

        q_interp = self._anim_q_start + \
            (self._anim_q_target - self._anim_q_start) * t_eased
        self.robot.q = q_interp

        for i, a in enumerate(q_interp):
            self.control_win.set_joint_display(i, np.degrees(a))

        self.update_plot()

        if self._anim_step >= self._anim_steps:
            self._anim_timer.stop()
            # Snap exactly to target to avoid floating-point drift
            self.robot.q = self._anim_q_target
            for i, a in enumerate(self._anim_q_target):
                self.control_win.set_joint_display(i, np.degrees(a))
            self.update_plot()

    def reset_robot(self):
        self.robot.q = np.zeros(self.robot.n)
        for i in range(self.robot.n):
            self.control_win.set_joint_display(i, 0)
        self.update_plot()

    def close_program(self):
        QApplication.quit()
        sys.exit()

    def update_plot(self):
        # Perform FK — T is a list/array of SE3, one per joint frame
        # (base -> end effector), same length as before.
        T = self.robot.fkine_all(self.robot.q)

        # Single pass over T to grab both position and the world-frame
        # rotation axis (z-axis of each transform's rotation matrix),
        # instead of two separate list comprehensions.
        n = len(T)
        xyz = np.empty((n, 3))
        axes = np.empty((n, 3))
        for i, x in enumerate(T):
            xyz[i] = x.t
            axes[i] = x.R[:, 2]

        # Update 3D Plot (cylinders, replacing the old flat line)
        self.plot_win.update_view(xyz, axes)

        # Update Table in Control Panel
        self.control_win.update_coordinate_table(xyz)
