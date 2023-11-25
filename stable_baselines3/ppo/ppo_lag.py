from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import torch as th
import numpy as np

from stable_baselines3.common.buffers import RolloutBuffer, RolloutCostBuffer
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCostPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.lagrange import Lagrange
from ppo import PPO

SelfPPOLAG = TypeVar("SelfPPOLAG", bound="PPOLag")


class PPOLag(PPO):
    def __init__(
        self,
        policy: type[ActorCriticCostPolicy],
        env: GymEnv | str,
        lagrange_cfgs: dict,
        learning_rate: float | Schedule = 0.0003,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float | Schedule = 0.2,
        clip_range_vf: float | Schedule | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: type[RolloutCostBuffer] | None = None,
        rollout_buffer_kwargs: Dict[str, Any] | None = None,
        target_kl: float | None = None,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: Dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        assert type(policy) is ActorCriticCostPolicy, "only support ActorCriticCostPolicy"
        super().__init__(
            policy,
            env,
            learning_rate,
            n_steps,
            batch_size,
            n_epochs,
            gamma,
            gae_lambda,
            clip_range,
            clip_range_vf,
            normalize_advantage,
            ent_coef,
            vf_coef,
            max_grad_norm,
            use_sde,
            sde_sample_freq,
            rollout_buffer_class,
            rollout_buffer_kwargs,
            target_kl,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
        )
        self.lagrange_cfgs = lagrange_cfgs

    def _setup_model(self) -> None:
        super()._setup_model()
        self._lagrange: Lagrange = Lagrange(**self.lagrange_cfgs)

    def train(self) -> None:
        return super().train()

    def collect_rollouts(
        self, env: VecEnv, callback: BaseCallback, rollout_buffer: RolloutBuffer, n_rollout_steps: int
    ) -> bool:
        return super().collect_rollouts(env, callback, rollout_buffer, n_rollout_steps)

    def learn(
        self: SelfPPOLAG,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPOLAG",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPOLAG:
        return super().learn(total_timesteps, callback, log_interval, tb_log_name, reset_num_timesteps, progress_bar)
