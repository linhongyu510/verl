# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""`apply_kl_penalty` reports the KL that drives the adaptive controller.

Aborted rollouts (``response_length == 0``, tracked as ``aborted_mask`` in
``metric_utils``) have an all-zero response mask. They must not be averaged in as
"zero KL", otherwise the reported value sits below the true one and the controller
lowers beta precisely when aborts are frequent.
"""

import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.ppo.core_algos import AdaptiveKLController, FixedKLController
from verl.trainer.ppo.ray_trainer import apply_kl_penalty


def _make_data(response_mask: torch.Tensor, kl_per_token: float):
    """Build a batch whose per-token KL equals `kl_per_token` everywhere."""
    batch_size, response_length = response_mask.shape
    old_log_probs = torch.full((batch_size, response_length), kl_per_token, dtype=torch.float32)
    ref_log_prob = torch.zeros((batch_size, response_length), dtype=torch.float32)
    return DataProto(
        batch=TensorDict(
            {
                "response_mask": response_mask,
                "token_level_scores": torch.zeros((batch_size, response_length), dtype=torch.float32),
                "old_log_probs": old_log_probs,
                "ref_log_prob": ref_log_prob,
            },
            batch_size=batch_size,
        )
    )


def test_aborted_rollouts_do_not_dilute_reported_kl():
    """Two aborted rows out of eight must not drag the mean down."""
    response_mask = torch.ones(8, 4, dtype=torch.long)
    response_mask[6:] = 0  # two aborted rollouts
    data = _make_data(response_mask, kl_per_token=0.4)

    _, metrics = apply_kl_penalty(data, kl_ctrl=FixedKLController(kl_coef=0.1), kl_penalty="kl")

    # Every non-aborted sequence has a per-token KL of exactly 0.4.
    assert metrics["actor/reward_kl_penalty"] == torch.tensor(0.4).item()


def test_reported_kl_matches_batch_without_the_aborted_rows():
    """The reported KL must not depend on how many aborted rows ride along."""
    full_mask = torch.ones(6, 3, dtype=torch.long)
    kl_without_aborted = apply_kl_penalty(
        _make_data(full_mask, kl_per_token=0.25),
        kl_ctrl=FixedKLController(kl_coef=0.1),
        kl_penalty="kl",
    )[1]["actor/reward_kl_penalty"]

    padded_mask = torch.ones(9, 3, dtype=torch.long)
    padded_mask[6:] = 0  # same six real rollouts plus three aborted ones
    kl_with_aborted = apply_kl_penalty(
        _make_data(padded_mask, kl_per_token=0.25),
        kl_ctrl=FixedKLController(kl_coef=0.1),
        kl_penalty="kl",
    )[1]["actor/reward_kl_penalty"]

    assert kl_with_aborted == kl_without_aborted


def test_all_aborted_batch_reports_zero_and_does_not_crash():
    """A batch where every rollout aborted has no KL to report."""
    data = _make_data(torch.zeros(4, 3, dtype=torch.long), kl_per_token=0.4)

    updated, metrics = apply_kl_penalty(data, kl_ctrl=FixedKLController(kl_coef=0.1), kl_penalty="kl")

    assert metrics["actor/reward_kl_penalty"] == 0.0
    assert torch.isfinite(updated.batch["token_level_rewards"]).all()


def test_aborted_rollouts_do_not_push_the_adaptive_coefficient_down():
    """The controller must hold beta steady when the true KL sits at the target."""
    target_kl = 0.4
    response_mask = torch.ones(8, 4, dtype=torch.long)
    response_mask[6:] = 0
    controller = AdaptiveKLController(init_kl_coef=0.2, target_kl=target_kl, horizon=10_000)

    for _ in range(50):
        apply_kl_penalty(
            _make_data(response_mask, kl_per_token=target_kl),
            kl_ctrl=controller,
            kl_penalty="kl",
        )

    # current_kl == target_kl means proportional_error == 0, so beta must not move.
    # float32 rounding in the KL computation leaves a ~1e-10 drift over 50 steps.
    assert controller.value == pytest.approx(0.2, rel=1e-6)

    # Without excluding aborted rows the reported KL would read 0.3 instead of 0.4,
    # a 25% shortfall that drives beta down instead of holding it.
    diluted = AdaptiveKLController(init_kl_coef=0.2, target_kl=target_kl, horizon=10_000)
    for _ in range(50):
        diluted.update(current_kl=0.3, n_steps=8)
    assert diluted.value < 0.2


def test_token_level_rewards_still_exclude_masked_positions():
    """Guard the surrounding behaviour: masked positions carry no penalty."""
    response_mask = torch.tensor([[1, 1, 0]], dtype=torch.long)
    data = _make_data(response_mask, kl_per_token=0.5)

    updated, _ = apply_kl_penalty(data, kl_ctrl=FixedKLController(kl_coef=1.0), kl_penalty="kl")

    rewards = updated.batch["token_level_rewards"][0]
    assert rewards[2].item() == 0.0  # masked position untouched
    assert rewards[0].item() < 0.0  # penalised positions carry -beta * kl
