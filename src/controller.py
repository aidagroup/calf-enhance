import numpy as np
from src.envs.underwaterdrone import (
    TOP_Y,
    DRONE_MASS,
    GRAVITY,
    MAX_F_LONG,
    MAX_F_LAT,
)


class UnderwaterDroneNominalController:
    def __init__(
        self,
        kp_y: float = 2.0,
        kd_y: float = 1.2,
        kp_x: float = 1.5,
        kd_x: float = 0.8,
    ) -> None:
        self.kp_y = kp_y
        self.kd_y = kd_y
        self.kp_x = kp_x
        self.kd_x = kd_x

    def get_action(self, obs):
        if len(obs.shape) == 1:
            x, y, cos_theta, sin_theta, v_x, v_y, _ = obs
        else:  # obs is a batch of observations
            x, y, cos_theta, sin_theta, v_x, v_y = (
                obs[:, i, np.newaxis] for i in range(6)
            )

        y_err = TOP_Y - y
        Fy = GRAVITY * DRONE_MASS + self.kp_y * y_err - self.kd_y * v_y

        x_ref = 0.0
        x_err = x_ref - x
        Fx = self.kp_x * x_err - self.kd_x * v_x

        F_long = cos_theta * Fx + sin_theta * Fy
        F_lat = -sin_theta * Fx + cos_theta * Fy

        F_long = np.clip(F_long, -MAX_F_LONG, MAX_F_LONG)
        F_lat = np.clip(F_lat, -MAX_F_LAT, MAX_F_LAT)

        return np.hstack([F_long, F_lat])
