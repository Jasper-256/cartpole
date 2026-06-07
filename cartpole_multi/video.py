from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from cartpole_multi.env import MultiPendulumCartPoleEnv


def record_policy_video(torch_model, obs_dim: int, args: argparse.Namespace) -> str:
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    video_path = video_dir / (
        f"cartpole_{args.num_pendulums}p_{args.total_timesteps}steps_seed{args.seed}.mp4"
    )

    env = MultiPendulumCartPoleEnv(num_pendulums=args.num_pendulums, seed=args.seed + 10_000)
    obs, _info = env.reset()
    frames = [render_env_frame(env, width=args.video_width, height=args.video_height)]

    torch_model.eval()
    device = next(torch_model.parameters()).device
    with torch.no_grad():
        for _step in range(args.video_steps):
            obs_tensor = torch.as_tensor(obs.reshape(1, obs_dim), dtype=torch.float32, device=device)
            logits, _value = torch_model(obs_tensor)
            action = int(torch.argmax(logits, dim=-1).item())
            obs, _reward, terminated, truncated, _info = env.step(action)
            frames.append(render_env_frame(env, width=args.video_width, height=args.video_height))
            if terminated or truncated:
                break

    write_video(frames, video_path, fps=args.video_fps)
    return str(video_path)


def render_env_frame(env: MultiPendulumCartPoleEnv, width: int = 800, height: int = 450) -> np.ndarray:
    p = env.params
    frame = Image.new("RGB", (width, height), (248, 249, 251))
    draw = ImageDraw.Draw(frame)

    track_y = int(height * 0.68)
    margin = 72
    world_width = 2 * p.x_threshold
    px_per_meter = (width - 2 * margin) / world_width
    cart_x = int(width / 2 + float(env.state[0]) * px_per_meter)
    cart_w = 70
    cart_h = 34
    pivot_y = track_y - cart_h // 2

    draw.line((margin, track_y, width - margin, track_y), fill=(44, 51, 63), width=3)
    stable_left = int(width / 2 - p.stable_x_threshold * px_per_meter)
    stable_right = int(width / 2 + p.stable_x_threshold * px_per_meter)
    draw.rectangle((stable_left, track_y + 8, stable_right, track_y + 15), fill=(109, 184, 138))

    cart_box = (
        cart_x - cart_w // 2,
        track_y - cart_h,
        cart_x + cart_w // 2,
        track_y,
    )
    draw.rounded_rectangle(cart_box, radius=6, fill=(45, 91, 145), outline=(23, 48, 77), width=2)
    draw.ellipse((cart_x - 28, track_y - 4, cart_x - 12, track_y + 12), fill=(38, 38, 38))
    draw.ellipse((cart_x + 12, track_y - 4, cart_x + 28, track_y + 12), fill=(38, 38, 38))

    theta = env.state[2 : 2 + env.num_pendulums]
    colors = [
        (222, 91, 73),
        (54, 137, 121),
        (132, 91, 184),
        (214, 151, 58),
        (66, 121, 196),
    ]
    pole_len_px = int(150 * min(1.0, 1.4 / max(env.num_pendulums, 1)))
    pivot_offsets = np.linspace(-12, 12, env.num_pendulums) if env.num_pendulums > 1 else [0]
    for idx, (angle, offset) in enumerate(zip(theta, pivot_offsets)):
        pivot_x = int(cart_x + offset)
        tip_x = int(pivot_x + pole_len_px * np.sin(float(angle)))
        tip_y = int(pivot_y - pole_len_px * np.cos(float(angle)))
        color = colors[idx % len(colors)]
        draw.line((pivot_x, pivot_y, tip_x, tip_y), fill=color, width=6)
        draw.ellipse((pivot_x - 5, pivot_y - 5, pivot_x + 5, pivot_y + 5), fill=(24, 31, 42))
        draw.ellipse((tip_x - 9, tip_y - 9, tip_x + 9, tip_y + 9), fill=color)

    stable_text = "stable" if env.is_stable() else "unstable"
    draw.text((18, 16), f"step {env.step_count}  {stable_text}", fill=(24, 31, 42))
    draw.text((18, 38), f"x={float(env.state[0]):+.2f}", fill=(24, 31, 42))
    return np.asarray(frame, dtype=np.uint8)


def write_video(frames: list[np.ndarray], video_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        gif_path = video_path.with_suffix(".gif")
        images = [Image.fromarray(frame) for frame in frames]
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        video_path.write_text(f"ffmpeg not found; wrote {gif_path.name} instead\n")
        return

    height, width, _channels = frames[0].shape
    cmd = [
        ffmpeg,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vcodec",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(video_path),
    ]
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdin is not None
    for frame in frames:
        process.stdin.write(frame.astype(np.uint8, copy=False).tobytes())
    _stdout, stderr = process.communicate()
    if process.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")


def open_video(video_path: str) -> None:
    if os.name != "posix":
        return

    opener = "open" if os.uname().sysname == "Darwin" else "xdg-open"
    try:
        subprocess.Popen(
            [opener, video_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        print(f"could_not_open_video={exc}")
