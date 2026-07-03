from pathlib import Path

def move_legacy_frame_plots_up(data_dir, overwrite=False) -> None:
    """Temporary helper: move old scan_*/frames/frame_*_fit.png files up one level.
    args:
        data_dir: directory containing scan_*/frames folders
        overwrite: whether to overwrite existing files, if a frame plot already exists in the scan_*/ folder"""
    export_root = Path(data_dir) / "sin2psi_export"
    if not export_root.exists():
        raise FileNotFoundError(f"No sin2psi_export folder found in {data_dir}")

    moved = 0
    skipped = 0
    removed_dirs = 0

    for frames_dir in sorted(export_root.glob("scan_*/frames")):
        scan_dir = frames_dir.parent
        for frame_path in sorted(frames_dir.glob("frame_*_fit.png")):
            target = scan_dir / frame_path.name
            if target.exists() and not overwrite:
                print(f"Skipping existing file: {target}")
                skipped += 1
                continue
            if target.exists() and overwrite:
                target.unlink()
            frame_path.rename(target)
            moved += 1

        try:
            frames_dir.rmdir()
            removed_dirs += 1
        except OSError:
            print(f"Leaving non-empty frames folder: {frames_dir}")

    print(f"Moved {moved} frame plot(s), skipped {skipped}, removed {removed_dirs} empty frames folder(s).")