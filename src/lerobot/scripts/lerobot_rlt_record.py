# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License").

"""Record labeled rollouts from either the initial ACP+RTC or an RLT export."""

from lerobot.configs import parser
from lerobot.scripts.lerobot_record import RecordConfig, record
from lerobot.scripts.recording_rlt_resume import record_to_target
from lerobot.utils.import_utils import register_third_party_plugins


@parser.wrap()
def rlt_record(cfg: RecordConfig):
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
    # Reset is performed by the operator through the recorder's existing reset
    # phase; this entrypoint does not initiate an automatic robot reset motion.
    return record_to_target(cfg, record)


def main():
    register_third_party_plugins()
    rlt_record()


if __name__ == "__main__":
    main()
