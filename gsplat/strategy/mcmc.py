import math
from dataclasses import dataclass
from typing import Any, Dict, Union

import torch
from torch import Tensor

from .base import Strategy
from .ops import inject_noise_to_position, relocate, sample_add

try:
    from typing import Literal
except:
    from typing_extensions import Literal

@dataclass
class MCMCStrategy(Strategy):
    """Strategy that follows the paper:

    `3D Gaussian Splatting as Markov Chain Monte Carlo <https://arxiv.org/abs/2404.09591>`_

    This strategy will:

    - Periodically teleport GSs with low opacity to a place that has high opacity.
    - Periodically introduce new GSs sampled based on the opacity distribution.
    - Periodically perturb the GSs locations.

    Args:
        cap_max (int): Maximum number of GSs. Default to 1_000_000.
        noise_lr (float): MCMC samping noise learning rate. Default to 5e5.
        refine_start_iter (int): Start refining GSs after this iteration. Default to 500.
        refine_stop_iter (int): Stop refining GSs after this iteration. Default to 25_000.
        refine_every (int): Refine GSs every this steps. Default to 100.
        min_opacity (float): GSs with opacity below this value will be pruned. Default to 0.005.
        verbose (bool): Whether to print verbose information. Default to False.

    Examples:

        >>> from gsplat import MCMCStrategy, rasterization
        >>> params: Dict[str, torch.nn.Parameter] | torch.nn.ParameterDict = ...
        >>> optimizers: Dict[str, torch.optim.Optimizer] = ...
        >>> strategy = MCMCStrategy()
        >>> strategy.check_sanity(params, optimizers)
        >>> strategy_state = strategy.initialize_state()
        >>> for step in range(1000):
        ...     render_image, render_alpha, info = rasterization(...)
        ...     loss = ...
        ...     loss.backward()
        ...     strategy.step_post_backward(params, optimizers, strategy_state, step, info, lr=1e-3)

    """

    cap_max: int = 6_000_000
    noise_lr: float = 5e5
    refine_start_iter: int = 3000
    refine_stop_iter: int = 100_000
    refine_every: int = 100
    min_opacity: float = 0.005
    verbose: bool = False
    densify_portion: float = 0.001  # 每次 densify 增加的点数比例

    schedule_mode: Literal["cosine", "linear", "exp", "staged", "original"] = "cosine"
    densify_step_portion_max: float = 0.005     # 单次 refine 最多按当前点数的增长比例
    densify_step_min_add: int = 0               # 单次最少新增个数（可为 0）

    def initialize_state(self) -> Dict[str, Any]:
        """Initialize and return the running state for this strategy."""
        n_max = 51
        binoms = torch.zeros((n_max, n_max))
        print("[MCMC] Densify schedule:", self.schedule_mode)
        for n in range(n_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)
        return {"binoms": binoms,
                "n0": None,              # 初始化点数，首次调用时读取
                "last_target_n": None,   # 上次的目标点数，保证单调递增
                }

    def check_sanity(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
    ):
        """Sanity check for the parameters and optimizers.

        Check if:
            * `params` and `optimizers` have the same keys.
            * Each optimizer has exactly one param_group, corresponding to each parameter.
            * The following keys are present: {"means", "scales", "quats", "opacities"}.

        Raises:
            AssertionError: If any of the above conditions is not met.

        .. note::
            It is not required but highly recommended for the user to call this function
            after initializing the strategy to ensure the convention of the parameters
            and optimizers is as expected.
        """

        super().check_sanity(params, optimizers)
        # The following keys are required for this strategy.
        for key in ["means", "scales", "quats", "opacities"]:
            assert key in params, f"{key} is required in params but missing."

    # def step_pre_backward(
    #     self,
    #     params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
    #     optimizers: Dict[str, torch.optim.Optimizer],
    #     # state: Dict[str, Any],
    #     step: int,
    #     info: Dict[str, Any],
    # ):
    #     """Callback function to be executed before the `loss.backward()` call."""
    #     pass

    def step_post_backward(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        info: Dict[str, Any],
        lr: float,
    ):
        """Callback function to be executed after the `loss.backward()` call.

        Args:
            lr (float): Learning rate for "means" attribute of the GS.
        """
        # move to the correct device
        state["binoms"] = state["binoms"].to(params["means"].device)

        binoms = state["binoms"]

        # 读取初始化点数 N0（只做一次）
        if state["n0"] is None:
            state["n0"] = int(params["means"].shape[0])
            state["last_target_n"] = state["n0"]
            if self.verbose:
                print(f"[MCMC] N0 (initial GS count) = {state['n0']}")

        if (
            step < self.refine_stop_iter
            and step > self.refine_start_iter
            and step % self.refine_every == 0
        ):
            # teleport GSs
            n_relocated_gs = self._relocate_gs(params, optimizers, binoms)
            if self.verbose:
                print(f"Step {step}: Relocated {n_relocated_gs} GSs.")

            # add new GSs
            if self.schedule_mode == "cosine":
                n_new_gs = self._scheduled_add(params, optimizers, state, step, binoms)
            elif self.schedule_mode == "original":
                n_new_gs = self._add_new_gs(params, optimizers, binoms)
            if self.verbose:
                print(
                    f"Step {step}: Added {n_new_gs} GSs. "
                    f"Now having {len(params['means'])} GSs."
                )

            torch.cuda.empty_cache()

        # # 3) 位置噪声注入：随 densify 进度退火（后期更小）
        # progress = self._progress(step)  # [0,1]
        # 这里乘以 (1 - progress) 做线性退火
        # noise_scale = lr * self.noise_lr * (1.0 - progress)
        # add noise to GSs
        inject_noise_to_position(
            params=params, optimizers=optimizers, state={}, scaler=lr * self.noise_lr
        )

    # ------------------------ 调度核心 ------------------------

    def _progress(self, step: int) -> float:
        """归一化 densify 进度 p ∈ [0,1]。在窗口外返回 0 或 1。"""
        if step <= self.refine_start_iter:
            return 0.0
        if step >= self.refine_stop_iter:
            return 1.0
        span = max(1, self.refine_stop_iter - self.refine_start_iter)
        return float(step - self.refine_start_iter) / float(span)

    def _schedule_fraction(self, p: float) -> float:
        """给定进度 p，返回目标“增量完成度” f ∈ [0,1]。"""
        p = max(0.0, min(1.0, p))
        mode = (self.schedule_mode or "cosine").lower()

        if mode == "linear":
            return p
        elif mode == "cosine":
            # ease-in-out：前期快、后期慢
            return 0.5 - 0.5 * math.cos(math.pi * p)
        elif mode == "exp":
            # 早期更快，后期更慢；k>0 越大越前倾
            k = 3.0
            return (math.exp(k * p) - 1.0) / (math.exp(k) - 1.0)
        elif mode == "staged":
            # 分段线性插值
            knots = sorted(self.staged_knots, key=lambda x: x[0])
            # 边界
            if p <= knots[0][0]:
                return knots[0][1]
            if p >= knots[-1][0]:
                return knots[-1][1]
            # 区间内线性插值
            for (p0, f0), (p1, f1) in zip(knots[:-1], knots[1:]):
                if p0 <= p <= p1:
                    t = (p - p0) / max(1e-12, (p1 - p0))
                    return f0 + t * (f1 - f0)
            return knots[-1][1]
        else:
            # 未知模式，退回线性
            return p

    @torch.no_grad()
    def _scheduled_add(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        state: Dict[str, Any],
        step: int,
        binoms: Tensor,
    ) -> int:
        """根据调度曲线把点数推向目标总量，按差额补齐并限幅。"""
        n0 = int(state["n0"])
        cur = int(params["means"].shape[0])
        cap = int(self.cap_max)

        # 计算目标：N(t) = N0 + (cap - N0) * f(progress)
        p = self._progress(step)
        frac = self._schedule_fraction(p)
        target = int(round(n0 + (cap - n0) * frac))

        # 单调性保护（避免浮点/四舍五入造成回退）
        if state["last_target_n"] is not None:
            target = max(target, int(state["last_target_n"]))
        state["last_target_n"] = target

        # 需要新增的点数（差额）
        need = max(0, min(target - cur, cap - cur))
        if need <= 0:
            return 0

        # 每步上限：按当前点数的比例限幅，且保证至少 min_add
        step_cap = max(self.densify_step_min_add, int(cur * self.densify_step_portion_max))
        n_add = int(min(need, step_cap))

        if n_add > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_add,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_add


    @torch.no_grad()
    def _relocate_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        opacities = torch.sigmoid(params["opacities"].flatten())
        dead_mask = opacities <= self.min_opacity
        n_gs = dead_mask.sum().item()
        if n_gs > 0:
            relocate(
                params=params,
                optimizers=optimizers,
                state={},
                mask=dead_mask,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs

    @torch.no_grad()
    def _add_new_gs(
        self,
        params: Union[Dict[str, torch.nn.Parameter], torch.nn.ParameterDict],
        optimizers: Dict[str, torch.optim.Optimizer],
        binoms: Tensor,
    ) -> int:
        current_n_points = len(params["means"])

        n_target = min(self.cap_max, int((1.0 + self.densify_portion) * current_n_points))
        n_gs = max(0, n_target - current_n_points)
        if n_gs > 0:
            sample_add(
                params=params,
                optimizers=optimizers,
                state={},
                n=n_gs,
                binoms=binoms,
                min_opacity=self.min_opacity,
            )
        return n_gs
