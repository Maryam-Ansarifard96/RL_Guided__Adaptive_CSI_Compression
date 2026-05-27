from collections import deque
COMPRESSION_RATIOS = [1/4, 1/8, 1/16, 1/32]
class ATHController:
    def __init__(
        self,
        theta_H=15.0,
        theta_M=5.0,
        theta_N=0.15,
        theta_delta=0.02,
        T_h=5,
        T_f=5,
        ema_lambda=0.1,
        compression_list=COMPRESSION_RATIOS
    ):
        self.theta_H = theta_H
        self.theta_M = theta_M
        self.theta_N = theta_N
        self.theta_delta = theta_delta
        self.T_h = T_h
        self.T_f = T_f
        self.ema_lambda = ema_lambda
        self.compression_list = compression_list

        self.reset()

    def reset(self):
        self.current_action = 0
        self.target_action = 0
        self.mode_counter = 0
        self.nmse_ema = None
        self.delta_counter = 0
        self.nmse_hist = deque(maxlen=20)

    def select_action(self, snr_db, nmse):
        # ---- NMSE EMA ----
        if self.nmse_ema is None:
            self.nmse_ema = nmse
        else:
            self.nmse_ema = (1 - self.ema_lambda) * self.nmse_ema + self.ema_lambda * nmse

        self.nmse_hist.append(self.nmse_ema)

        delta_nmse = 0.0
        if len(self.nmse_hist) >= 2:
            delta_nmse = self.nmse_hist[-1] - self.nmse_hist[0]

        # ---- SINR-based decision ----
        if snr_db > self.theta_H:
            desired_action = 1  # highest ratio
        elif snr_db > self.theta_M:
            desired_action = 2
        else:
            desired_action = 3

        # ---- NMSE override ----
        if nmse > self.theta_N:
            desired_action = 1

        # ---- Hysteresis ----
        if desired_action != self.current_action:
            if desired_action == self.target_action:
                self.mode_counter += 1
            else:
                self.target_action = desired_action
                self.mode_counter = 1

            if self.mode_counter >= self.T_h:
                self.current_action = self.target_action
                self.mode_counter = 0
        else:
            self.mode_counter = 0

        # ---- Fine-tune trigger ----
        finetune = False
        if delta_nmse > self.theta_delta:
            self.delta_counter += 1
        else:
            self.delta_counter = 0

        if self.delta_counter >= self.T_f:
            finetune = True
            self.delta_counter = 0

        # action = 0 means "do nothing"
        return self.current_action if finetune else 0