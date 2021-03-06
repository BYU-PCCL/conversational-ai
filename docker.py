#!/usr/bin/env python3
"""Starts the conversational-ai docker container."""
import datetime
import getpass
import multiprocessing as mp
import os
import platform
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Union

_Path = Union[str, Path]


def run(
    image: str,
    name: str,
    command: Union[List[str], str, None] = None,
    volumes: Dict[_Path, _Path] = {"/mnt": "/mnt"},  # noqa: B006
    tty: bool = False,
    pull: bool = False,
    follow_logs: bool = False,
    rm: bool = False,
    args: List[str] = [  # noqa: B006
        "--network=host",
        "--ipc=host",
        "--shm-size=8g",
        "--ulimit=memlock=-1",
        "--ulimit=stack=67108864",
        f"--cpus={max(int(0.33 * mp.cpu_count()), 8)}",
        f"--user={os.getuid()}:{os.getgid()}",
    ],
    **kwargs,  # not used; just here so we can pass in whatever we get from the CLI
) -> None:
    """Start the container."""
    cmd = [] if not command else command
    if isinstance(cmd, str):
        cmd = shlex.split(cmd)

    if pull:
        subprocess.run(["docker", "pull", image], stdout=sys.stdout, stderr=sys.stderr)
        print("")

    args = ["docker", "run", f"--name={name}", "-it" if tty else "--detach"] + args
    if rm:
        args.append("--rm")

    gpus = os.getenv("NVIDIA_VISIBLE_DEVICES", os.getenv("NV_GPU", "all"))
    args.append(f"--env=NVIDIA_VISIBLE_DEVICES={gpus}")
    # handle compatability with older versions of `nvidia-container-toolkit`
    result = subprocess.check_output(["docker", "run", "--help"], encoding="utf8")
    args.append(f"--gpus={gpus}" if "--gpus" in result else "--runtime=nvidia")

    for src, dst in filter(lambda p: Path(p[0]).exists(), volumes.items()):
        args.append(f"--volume={Path(src).absolute()}:{dst}")

    subprocess.run(
        args + [image] + cmd,
        stdin=sys.stdin if tty else None,
        stdout=sys.stdout if tty else None,
        check=True,
    )

    if follow_logs and not tty:
        print(name, "(Press CTRL+C to stop viewing the logs)", sep="\n\n")
        subprocess.run(["docker", "logs", "-f", name])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=True,
        epilog="All other args will be passed to the container's process",
    )

    parser.add_argument(
        "--image",
        help="Docker image to use for the container",
        default="pccl/conversational-ai:latest",
    )
    parser.add_argument(
        "--name",
        help="Container name; formatted with values if available",
        default="cai_{timestamp}",
    )
    parser.add_argument(
        "-m",
        "--module",
        help="The module for python to run inside the container",
        default="conversational_ai.t5_model",
    )
    parser.add_argument(
        "-t",
        "--tty",
        help="Run container in the foreground, allocating an interactive pseudo-TTY",
        action="store_true",
    )
    parser.add_argument(
        "--pull",
        help="Pull the image before running the container",
        action="store_true",
    )
    parser.add_argument(
        "--rm",
        help="Automatically remove the container when it exits",
        action="store_true",
    )
    parser.add_argument(
        "--follow-logs",
        help="Follow log output (only if tty was not specified)",
        action="store_true",
    )
    # TODO: add `--volumes` flag

    args, extra_args = parser.parse_known_args()

    # hardcode the tz for now because some servers are in random timezones
    tz = datetime.timezone(-datetime.timedelta(hours=6))
    name = args.name.format(
        image=args.image,
        username=getpass.getuser(),
        hostname=platform.node(),
        timestamp=datetime.datetime.now(tz=tz).isoformat(timespec="milliseconds"),
    )
    # TODO: handle docker container name must start with an alphanumeric character?
    name = re.sub(r"[^\d\w_.-]+", "-", name)

    run_kwargs = {
        **vars(args),
        "name": name,
        "command": shlex.split(args.module),
        "volumes": {
            "./checkpoints": "/workspace/checkpoints/",
            "./config": "/workspace/config/",
            "./data": "/workspace/data/",
            "./chats": "/workspace/chats/",
            "/mnt": "/mnt",
        },
    }

    run_kwargs["command"].extend(extra_args)

    try:
        run(**run_kwargs)
    except KeyboardInterrupt:
        sys.exit()
