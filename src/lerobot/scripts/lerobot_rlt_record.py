# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Record labeled rollouts from either the initial ACP+RTC or an RLT export."""

from dataclasses import dataclass

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.scripts.recording_rlt_control import RLTManualEpisodeController
from lerobot.scripts.recording_rlt_resume import record_to_target
from lerobot.utils.import_utils import register_third_party_plugins


@dataclass
class RLTRecordConfig(RecordConfig):
    # Duration of the operator-requested return to this command's startup pose.
    reset_duration_s: float = 5.0


@parser.wrap()
def rlt_record(cfg: RLTRecordConfig):
    if not cfg.remote_policy.enable or not cfg.remote_policy.training_time_rtc:
        raise ValueError(
            "RLT rollout recording requires remote_policy.enable=true and training_time_rtc=true."
        )
    if cfg.remote_policy.policy_type not in {"pi05", "rlt"}:
        raise ValueError("Use remote_policy.policy_type=pi05 for round 0 or rlt for subsequent rounds.")
    cfg.enable_episode_outcome_labeling = True
    cfg.require_episode_success_label = True
    # Timeouts and early termination must never become positive demonstrations.
    cfg.default_episode_success = "failure"
    cfg.enable_collector_policy_id = True
    cfg.collector_policy_id_policy = (
        cfg.collector_policy_id_policy or cfg.remote_policy.pretrained_name_or_path
    )
    cfg.policy_sync_to_teleop = cfg.teleop is not None
    cfg.intervention_state_machine_enabled = cfg.teleop is not None
    if any(
        key.lower() in {"r", "t"}
        for key in (cfg.intervention_toggle_key, cfg.episode_success_key, cfg.episode_failure_key)
    ):
        raise ValueError("r and t are reserved for RLT reset/start controls")
    # Each s/f must finish encoding and flush all metadata before r/t can act.
    cfg.dataset.video_encoding_batch_size = 1
    controller = RLTManualEpisodeController(cfg.reset_duration_s)
    cfg._rlt_episode_controller = controller
    cfg._on_record_connected = controller.on_record_connected
    # The shared parser accepts only an exact RecordConfig type. This subclass
    # is already parsed; avoid reparsing CLI and losing its controller hooks.
    return record_to_target(cfg, record.__wrapped__)


def main():
    register_third_party_plugins()
    rlt_record()


if __name__ == "__main__":
    main()
