"""Launch/query one RLT training round on Ubuntu; no robot control or recording."""

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
DATA_ROOT = "/nfs/baiyuntian/assets/rlt_0909"
MODEL_ROOT = "/nfs/baiyuntian/assets/models/banknote_rlt"


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def make_plan(args):
    if args.round < 0 or args.gpu < 0 or args.episodes < 1 or args.token_steps < 1 or args.ac_steps < 2:
        raise ValueError("Require nonnegative round/GPU, positive episodes/token steps, and ac-steps >= 2")
    config = json.loads((REPO / "configs/rlt/windows_record.json").read_text(encoding="utf-8"))
    model_root = Path(args.model_root).resolve()
    dataset = Path(args.data_root).resolve() / f"banknote_round_{args.round}"
    base = config["remote_policy"]["pretrained_name_or_path"]
    source_policy = base if args.round == 0 else str(model_root / f"round_{args.round}")
    output = model_root / f"round_{args.round + 1}"
    job = model_root / "jobs" / f"from_round_{args.round}"
    specs = [{"repo_id": f"local/banknote_round_{args.round}", "root": str(dataset), "source": "rollout"}]
    if args.round == 0:
        initial = json.loads((REPO / "configs/rlt/initial.json").read_text(encoding="utf-8"))
        specs = [entry for entry in initial if entry["source"] == "demo"] + specs
        if not any(entry["source"] == "demo" for entry in specs):
            raise ValueError("Round 0 requires original demonstrations in configs/rlt/initial.json")
    new_cache = model_root / ("cache_initial" if args.round == 0 else f"cache_round_{args.round}")
    caches = [model_root / "cache_initial"] + [
        model_root / f"cache_round_{i}" for i in range(1, args.round + 1)
    ]
    entry = [sys.executable, "-u", "-m", "lerobot.scripts.lerobot_rlt_train"]
    commands = [
        (
            "cache",
            entry
            + [
                "cache",
                "--base-policy",
                base,
                "--dataset-spec",
                str(job / "dataset_spec.json"),
                "--output",
                str(new_cache),
                "--chunk-size",
                "50",
                "--rtc-delay",
                "21",
                "--task",
                config["dataset"]["single_task"],
                "--device",
                "cuda:0",
            ],
        )
    ]
    if args.round == 0:
        commands.append(
            (
                "token",
                entry
                + [
                    "token",
                    "--cache",
                    str(new_cache),
                    "--output",
                    str(model_root / "token"),
                    "--steps",
                    str(args.token_steps),
                    "--device",
                    "cuda:0",
                ],
            )
        )
    ac = entry + ["ac"]
    for cache in caches:
        ac += ["--cache", str(cache)]
    if args.round == 0:
        ac += ["--token-checkpoint", str(model_root / "token/token.pt"), "--residual-scale", "0.1"]
    else:
        ac += ["--policy", source_policy, "--resume"]
    ac += ["--output", str(output), "--steps", str(args.ac_steps), "--device", "cuda:0"]
    commands.append(("actor-critic", ac))
    return {
        "dataset": str(dataset),
        "source_policy": source_policy,
        "output": str(output),
        "job": str(job),
        "specs": specs,
        "cache": str(new_cache),
        "caches": [str(path) for path in caches],
        "commands": commands,
    }


def check_checkpoint(output, plan=None, ac_steps=None):
    output = Path(output)
    for name in ("config.json", "rlt_model.pt", "trainer_state.pt", "training_manifest.json"):
        if not (output / name).is_file() or (output / name).stat().st_size == 0:
            raise ValueError(f"Incomplete checkpoint: {output / name}")
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    if not (config.get("type") == "rlt" and config.get("actor_ready") and config.get("token_ready")):
        raise ValueError(f"Checkpoint is not ready for RLT deployment: {output}")
    if config.get("chunk_size") != 50 or config.get("rtc_prefix_steps") != 21:
        raise ValueError("Checkpoint RTC contract must be H=50, d=21")
    if plan is not None:
        manifest = json.loads((output / "training_manifest.json").read_text(encoding="utf-8"))
        if manifest["caches"] != plan["caches"] or manifest["training_args"]["steps"] != ac_steps:
            raise ValueError("Checkpoint does not match the requested training round")


def run_stage(command):
    subprocess.run(command, check=True)


def worker(job):
    import fcntl

    from lerobot.scripts.lerobot_rlt_verify import verify_transfer

    job = Path(job).resolve()
    args = argparse.Namespace(**json.loads((job / "request.json").read_text(encoding="utf-8")))
    plan = make_plan(args)
    state = {
        "state": "RUNNING",
        "stage": "verify-upload",
        "pid": os.getpid(),
        "started_at": time.time(),
        "input_round": args.round,
        "physical_gpu": args.gpu,
        "output": plan["output"],
    }
    write_json(job / "status.json", state)
    try:
        # Protect cumulative caches/model outputs from concurrent round jobs.
        with (job.parent / ".training.lock").open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            report = verify_transfer(plan["dataset"], args.round, args.episodes, plan["source_policy"])
            if report["fps"] != 30:
                raise ValueError("This banknote training workflow requires 30 fps")
            if args.round > 0:
                check_checkpoint(plan["source_policy"])
                for cache in plan["caches"][:-1]:
                    manifest = json.loads((Path(cache) / "manifest.json").read_text(encoding="utf-8"))
                    if manifest.get("complete") is not True:
                        raise ValueError(f"Previous cache is incomplete: {cache}")
            write_json(job / "dataset_spec.json", plan["specs"])
            for stage, command in plan["commands"]:
                state["stage"] = stage
                write_json(job / "status.json", state)
                print(f"STAGE={stage}\n{shlex.join(command)}", flush=True)
                run_stage(command)
            check_checkpoint(plan["output"], plan, args.ac_steps)
        state.update(state="DONE", stage="complete", finished_at=time.time(), exit_code=0)
        write_json(job / "status.json", state)
        print(f"DONE: {plan['output']}", flush=True)
        return 0
    except BaseException as exc:
        state.update(state="FAILED", error=str(exc), finished_at=time.time(), exit_code=1)
        write_json(job / "status.json", state)
        traceback.print_exc()
        return 1


def start(args):
    plan = make_plan(args)
    print(f"banknote_round_{args.round} -> round_{args.round + 1}; physical GPU {args.gpu}")
    for stage, command in plan["commands"]:
        print(f"{stage}: {shlex.join(command)}")
    if args.dry_run:
        print("DRY_RUN: no training started and no files written")
        return 0
    if os.name != "posix":
        raise ValueError("Run start on H200 Ubuntu through SSH")
    if not (Path(plan["dataset"]) / "rlt_integrity.json").is_file():
        raise ValueError("Upload the verified dataset and rlt_integrity.json before starting training")
    if Path(plan["output"]).exists() or Path(plan["cache"]).exists():
        raise FileExistsError(
            "Round output/cache already exists; preserve it and inspect the previous job before recovery"
        )
    if args.round == 0 and (Path(args.model_root) / "token").exists():
        raise FileExistsError("Token output already exists; inspect the previous initial training job")
    if args.round > 0:
        check_checkpoint(plan["source_policy"])
    job = Path(plan["job"])
    job.mkdir(parents=True, exist_ok=False)
    request = {key: value for key, value in vars(args).items() if key not in {"command", "dry_run"}}
    write_json(job / "request.json", request)
    write_json(job / "status.json", {"state": "QUEUED", "output": plan["output"]})
    env = os.environ.copy()
    env.update(
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        PYTHONPATH=str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", ""),
        PYTHONUNBUFFERED="1",
    )
    try:
        with (job / "train.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "lerobot.scripts.lerobot_rlt_round",
                    "worker",
                    "--job",
                    str(job),
                ],
                cwd=REPO,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        (job / "pid").write_text(str(process.pid), encoding="ascii")
    except Exception as exc:
        write_json(job / "status.json", {"state": "FAILED", "error": str(exc)})
        raise
    print(f"STARTED pid={process.pid}\nLog: {job / 'train.log'}")
    return 0


def status(args):
    job = Path(args.model_root) / "jobs" / f"from_round_{args.round}"
    if not (job / "status.json").is_file():
        print("NOT_STARTED")
        return 1
    state = json.loads((job / "status.json").read_text(encoding="utf-8"))
    if state["state"] in ("RUNNING", "QUEUED"):
        pid = state.get("pid") or ((job / "pid").read_text() if (job / "pid").exists() else None)
        if pid:
            cmdline = Path(f"/proc/{pid}/cmdline")
            command = cmdline.read_bytes().replace(b"\0", b" ").decode() if cmdline.exists() else ""
            if "lerobot.scripts.lerobot_rlt_round worker" not in command or str(job.resolve()) not in command:
                state.update(
                    state="FAILED",
                    error="Training process exited before recording completion; inspect train.log",
                )
    if state["state"] == "DONE":
        try:
            check_checkpoint(state["output"])
        except Exception as exc:
            state.update(state="FAILED", error=str(exc))
    print(json.dumps(state, indent=2))
    log_path = job / "train.log"
    if args.tail and log_path.exists():
        from collections import deque

        with log_path.open(encoding="utf-8", errors="replace") as log:
            print("".join(deque(log, maxlen=args.tail)), end="")
    return (
        0
        if state["state"] == "DONE" or (not args.require_done and state["state"] in ("QUEUED", "RUNNING"))
        else 1
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    launch = commands.add_parser("start")
    launch.add_argument(
        "--round", type=int, required=True, help="Input dataset round N; trains model round N+1"
    )
    launch.add_argument("--gpu", type=int, required=True, help="Physical H200 GPU index")
    launch.add_argument("--episodes", type=int, default=20)
    launch.add_argument("--token-steps", type=int, default=5000)
    launch.add_argument("--ac-steps", type=int, default=5000)
    launch.add_argument("--data-root", default=DATA_ROOT)
    launch.add_argument("--model-root", default=MODEL_ROOT)
    launch.add_argument("--dry-run", action="store_true")
    query = commands.add_parser("status")
    query.add_argument("--round", type=int, required=True)
    query.add_argument("--model-root", default=MODEL_ROOT)
    query.add_argument("--tail", type=int, default=20)
    query.add_argument("--require-done", action="store_true")
    internal = commands.add_parser("worker", help=argparse.SUPPRESS)
    internal.add_argument("--job", required=True)
    args = parser.parse_args()
    try:
        result = (
            worker(args.job)
            if args.command == "worker"
            else start(args)
            if args.command == "start"
            else status(args)
        )
    except Exception as exc:
        parser.exit(1, f"RLT_ROUND_ERROR: {exc}\n")
    raise SystemExit(result)


if __name__ == "__main__":
    main()
