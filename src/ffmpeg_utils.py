import subprocess


def get_ffmpeg_cmd(
    width: int,
    height: int,
    fps: int,
    output_path: str,
    use_nvenc_if_available: bool = True,
) -> list:
    """
    Returns the appropriate ffmpeg command list, prioritizing NVENC over CPU encoding if available.
    """
    encoder = "libx264"
    preset = "fast"
    extra_flags = ["-crf", "18"]

    if use_nvenc_if_available:
        print("Checking NVENC availability...")
        try:
            # Functional test: Attempt to encode a dummy frame
            # This catches driver mismatches that pure detection misses
            test_cmd = [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=black:s=64x64:d=0.1",
                "-c:v",
                "h264_nvenc",
                "-f",
                "null",
                "-",
            ]
            result = subprocess.run(test_cmd, capture_output=True, text=True)

            if result.returncode == 0:
                encoder = "h264_nvenc"
                preset = "p4"
                # NVENC often doesn't support CRF. We rely on the bitrate (-b:v) set below.
                extra_flags = []
                print(f"NVENC verified! Using hardware acceleration ({encoder}).")
            else:
                # Log the reason for failure (e.g. driver version)
                err_msg = result.stderr.strip() or "Unknown error"
                print(f"NVENC present but failed validation. Reason: {err_msg}")
                print("Falling back to CPU encoding (libx264).")
        except FileNotFoundError:
            print("FFmpeg not found! Video generation will fail.")

    cmd = (
        [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            encoder,
            "-b:v",
            "5M",  # Bitrate
            "-pix_fmt",
            "yuv420p",
            "-preset",
            preset,
        ]
        + extra_flags
        + [output_path]
    )
    return cmd
