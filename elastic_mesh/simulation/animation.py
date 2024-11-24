import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import Optional, Callable, List
import imageio
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


class AnimationManager:
    """
    Manages animation creation and export for the mesh simulation.
    """

    def __init__(self, fps: int = 30, quality: int = 95):
        self.fps = fps
        self.quality = quality
        self.frames: List[np.ndarray] = []

    def capture_frame(self, fig):
        """Capture a matplotlib figure as a frame."""
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.frames.append(frame)

    def save_gif(self, filename: str, progress_callback: Optional[Callable] = None):
        """Save captured frames as an optimized GIF."""
        if not self.frames:
            console.print("[red]No frames to save![/red]")
            return

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("[cyan]Saving GIF...", total=len(self.frames))

            # Save with optimization
            with imageio.get_writer(
                    filename,
                    mode='I',
                    fps=self.fps,
                    optimize=True,
                    quality=self.quality
            ) as writer:
                for frame in self.frames:
                    writer.append_data(frame)
                    progress.update(task, advance=1)
                    if progress_callback:
                        progress_callback(len(self.frames))

        console.print(f"[green]Animation saved to {filename}[/green]")

    def save_mp4(self, filename: str, progress_callback: Optional[Callable] = None):
        """Save captured frames as an MP4 video."""
        if not self.frames:
            console.print("[red]No frames to save![/red]")
            return

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)

        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn()
        ) as progress:
            task = progress.add_task("[cyan]Saving MP4...", total=len(self.frames))

            writer = imageio.get_writer(
                filename,
                fps=self.fps,
                quality=self.quality,
                codec='h264',
                pixelformat='yuv420p'
            )

            for frame in self.frames:
                writer.append_data(frame)
                progress.update(task, advance=1)
                if progress_callback:
                    progress_callback(len(self.frames))

            writer.close()

        console.print(f"[green]Video saved to {filename}[/green]")

    def create_side_by_side_comparison(
            self,
            frames1: List[np.ndarray],
            frames2: List[np.ndarray],
            filename: str,
            labels: Optional[tuple] = None
    ):
        """Create a side-by-side comparison video of two simulations."""
        if not frames1 or not frames2:
            console.print("[red]Both frame sets are required for comparison![/red]")
            return

        # Ensure same number of frames
        min_frames = min(len(frames1), len(frames2))
        frames1 = frames1[:min_frames]
        frames2 = frames2[:min_frames]

        # Get frame dimensions
        height, width = frames1[0].shape[:2]

        # Create comparison frames
        comparison_frames = []
        for f1, f2 in zip(frames1, frames2):
            # Create combined frame
            combined = np.zeros((height, width * 2, 3), dtype=np.uint8)
            combined[:, :width] = f1
            combined[:, width:] = f2

            # Add labels if provided
            if labels:
                imageio.plugins.pillow.write_text(
                    combined,
                    (10, height - 30),
                    labels[0],
                    color=(255, 255, 255)
                )
                imageio.plugins.pillow.write_text(
                    combined,
                    (width + 10, height - 30),
                    labels[1],
                    color=(255, 255, 255)
                )

            comparison_frames.append(combined)

        # Save comparison video
        self.frames = comparison_frames
        self.save_mp4(filename)

    def clear_frames(self):
        """Clear stored frames to free memory."""
        self.frames = []
        console.print("[yellow]Frames cleared from memory[/yellow]")