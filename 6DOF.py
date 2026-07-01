from roboticstoolbox import DHRobot, RevoluteDH, models
from PyQt6.QtWidgets import QApplication

import numpy as np
import sys

from Gui.controller import RobotControllerGUI


# =====================
# Robot Dimensions (m)
# =====================

robot = DHRobot(
    [
        RevoluteDH(d=0.36, a=0.00, alpha=-np.pi/2, qlim=[2*-np.pi, 2*np.pi]),
        RevoluteDH(d=0.00, a=0.00, alpha=np.pi/2,
                   qlim=[2*-np.pi/2.8, 2*np.pi/2.8]),
        RevoluteDH(d=0.45, a=0.05, alpha=np.pi/2,
                   qlim=[-2.5, 2.5]),  # Increased d & a
        RevoluteDH(d=0.00, a=0.35, alpha=-np.pi/2,
                   qlim=[-np.pi/2.6, np.pi/2.5]),

        RevoluteDH(d=0.00, a=0.05, alpha=-np.pi/2,
                   qlim=[-np.pi/2.5, np.pi/2.6]),  # Offset created

        RevoluteDH(d=0.30, a=0.00, alpha=-np.pi/2,       qlim=[-np.pi, np.pi]),
    ],
    name="1M Reach Robotic Arm Assistant"
)
# =====================
# Start GUI
# =====================

if __name__ == "__main__":

    q = np.zeros(robot.n)
    T = robot.fkine(q)

    print("Home Position:")
    print(T)
    print("Position:", T.t)

    app = QApplication(sys.argv)

    gui = RobotControllerGUI(robot)
    sys.exit(app.exec())
