from roboticstoolbox import DHRobot, RevoluteDH
from PyQt6.QtWidgets import QApplication

import numpy as np
import sys

from Gui.controller import RobotControllerGUI


# =====================
# Robot Model
# =====================

d1, a2, a3 = 0.45, 0.65, 0.35

robot = DHRobot(
    [
        RevoluteDH(
            d=d1,
            a=0,
            alpha=-np.pi/2,
            qlim=[-np.pi, np.pi]
        ),

        RevoluteDH(
            d=0,
            a=a2,
            alpha=0,
            qlim=[-np.pi/2, np.pi/2]
        ),

        RevoluteDH(
            d=0,
            a=a3,
            alpha=0,
            qlim=[-5*np.pi/6, 5*np.pi/6]
        )
    ],
    name="3R Desktop Arm"
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
