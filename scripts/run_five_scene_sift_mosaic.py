"""Backward-compatible SIFT-only wrapper for the unified five-scene CLI."""

from scripts.run_five_scene_mosaic import main as run_five_scene_mosaic


def main() -> None:
    run_five_scene_mosaic(default_matcher="sift")


if __name__ == "__main__":
    main()
