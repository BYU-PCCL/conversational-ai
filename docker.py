#!/usr/bin/env python3
"""Start the conversational-ai docker container."""
import os
import platform
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

_Path = Union[str, Path]


def main(
    image: str,
    name: str,
    command: Union[List[str], str, None] = None,
    volumes: Dict[_Path, _Path] = {"/mnt": "/mnt"},
    tty: bool = False,
    pull: bool = False,
    args: List[str] = [
        "--rm",
        "--network=host",
        "--ipc=host",
        "--shm-size=8g",
        "--ulimit=memlock=-1",
        "--ulimit=stack=67108864",
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

    args = ["docker", "run", f"--name={name}", "-it" if tty else "--detach"] + args

    gpus = os.getenv("NVIDIA_VISIBLE_DEVICES", "all")
    args.append(f"--env=NVIDIA_VISIBLE_DEVICES={gpus}")
    # handle compatability with older versions of `nvidia-container-toolkit`
    result = subprocess.run(
        ["docker", "run", "--help"], capture_output=True, check=True, encoding="utf8",
    )
    args.append(f"--gpus={gpus}" if "--gpus" in result.stdout else "--runtime=nvidia")

    for src, dst in filter(lambda p: Path(p[0]).exists(), volumes.items()):
        args.append(f"--volume={Path(src).absolute()}:{dst}")

    subprocess.run(
        args + [image] + cmd,
        stdin=sys.stdin if tty else None,
        stdout=sys.stdout if tty else None,
        capture_output=not tty,
    )

    if not tty:
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
        "-i",
        "--image",
        help="The name of the docker image",
        default="pccl/conversational-ai",
    )
    parser.add_argument("--tag", help="The docker tag", default="latest")
    parser.add_argument(
        "--name",
        help="Container name format string",
        default="{name}_{tag}_{hostname}_{timestamp}",
    )
    parser.add_argument(
        "-c",
        "--command",
        help="The command to run inside the container",
        default="python3 models.py",
    )
    parser.add_argument(
        "-t",
        "--tty",
        help="Run container in foreground, allocating an interactive pseudo-TTY",
        action="store_true",
    )
    parser.add_argument(
        "--pull",
        help="Pull the image before running the container",
        action="store_true",
    )

    # TODO: add `--volumes` flag

    args, extra_args = parser.parse_known_args()

    name = args.name.format(
        name=args.image.split("/")[-1],
        tag=args.tag,
        hostname=platform.node(),
        # docker container names cannot have `:` in them
        timestamp=datetime.now().strftime("%Y-%m-%dT%H_%M_%S.%f"),
    )

    main_kwargs = {
        **vars(args),
        "image": f"{args.image}:{args.tag}",
        "name": name,
        "command": shlex.split(args.command),
        "volumes": {
            "./checkpoints": "/workspace/checkpoints/",
            "./data": "/workspace/data/",
            "./chats": "/workspace/chats/",
            "/mnt": "/mnt",
        },
    }

    # TODO: don't assume checkpoint prefix is `./checkpoints/`
    model_dir = Path("./checkpoints", name).resolve().absolute()
    main_kwargs["command"].append(f"--gin_param=MtfModel.model_dir='{model_dir}'")

    # NB: add the extra_args in after the model_dir param so it can be overridden
    main_kwargs["command"].extend(extra_args)

    try:
        main(**main_kwargs)
    except KeyboardInterrupt:
        sys.exit()
