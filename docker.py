#!/usr/bin/env python3
"""Start the conversational-ai docker container."""
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Union

_Path = Union[str, Path]


def main(
    image: str,
    name: str,
    checkpoints_dir: _Path,
    volumes: Dict[_Path, _Path] = {
        "/mnt": "/mnt",
        "./chats": "/workspace/chats",
        "./checkpoints": "/workspace/checkpoints",
    },
    tty: bool = False,
    extra_args: List[str] = [],
) -> None:
    """Starts the container."""
    result = subprocess.run(
        ["git", "remote", "get-url", "origin"], capture_output=True, encoding="utf8"
    )
    if image.split("/")[-1].split(":")[0] not in result.stdout:
        # current working directory is not the project root, so pull the image
        subprocess.run(["docker", "pull", image], stdout=sys.stdout, stderr=sys.stderr)

    args = [
        "docker",
        "run",
        f"--name={name}",
        "-it" if tty else "--detach",
        "--rm",
        "--publish-all",
        "--ipc=host",
        "--shm-size=8g",
        "--ulimit=memlock=-1",
        "--ulimit=stack=67108864",
    ]

    gpus = os.getenv("NVIDIA_VISIBLE_DEVICES", "all")
    args.append(f"--env=NVIDIA_VISIBLE_DEVICES={gpus}")
    # handle compatability with older versions of `nvidia-container-toolkit`
    result = subprocess.run(
        ["docker", "run", "--help"], capture_output=True, check=True, encoding="utf8",
    )
    args.append(f"--gpus={gpus}" if "--gpus" in result.stdout else "--runtime=nvidia")

    # HACK: figure out a better way to do this...
    if "chat.py" not in extra_args:
        checkpoints_dir = Path(checkpoints_dir, name).absolute()
        Path(checkpoints_dir).mkdir(exist_ok=True, parents=True)
        # TODO: use gin-config instead
        args.append(f"--env=CONVERSATIONAL_AI_MODEL_DIR={checkpoints_dir}")

    for local, container_path in filter(lambda p: Path(p[0]).exists(), volumes.items()):
        args.append(f"--volume={Path(local).absolute()}:{container_path}")

    subprocess.run(
        args + [image] + extra_args,
        stdin=sys.stdin if tty else None,
        stdout=sys.stdout if tty else None,
        capture_output=not tty,
    )

    if not tty:
        print(name, "(Press CTRL+C to stop viewing the logs)", sep="\n\n")
        subprocess.run(["docker", "logs", "-f", name])


if __name__ == "__main__":
    import argparse

    formatter = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=formatter)

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
        "--checkpoints",
        help="The directory root/prefix containing all checkpoint directories",
        type=Path,
        default=Path("/mnt/pccfs/not_backed_up/will/checkpoints/"),
    )
    parser.add_argument(
        "-t",
        "--tty",
        help="Run container in foreground, allocating an interactive pseudo-TTY",
        action="store_true",
    )

    args, extra_args = parser.parse_known_args()

    name = args.name.format(
        name=args.image.split("/")[-1],
        tag=args.tag,
        hostname=platform.node(),
        timestamp=int(datetime.now().timestamp()),
    )

    try:
        main(
            image=f"{args.image}:{args.tag}",
            name=name,
            checkpoints_dir=args.checkpoints,
            tty=args.tty,
            extra_args=extra_args,
        )
    except KeyboardInterrupt:
        sys.exit()
