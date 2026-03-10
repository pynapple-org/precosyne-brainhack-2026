from pathlib import Path


class Config:
    """Configuration for the brainhack package."""

    def __init__(self):
        self._root_folder = Path("/Volumes/Seagate Por/Pre-Cosyne-BrainHack-2026")

    @property
    def root_folder(self) -> Path:
        return self._root_folder

    def update(self, root_folder: str | Path | None = None):
        """Update configuration settings.

        Parameters
        ----------
        root_folder : str or Path, optional
            The root folder path for datasets.
        """
        if root_folder is not None:
            self._root_folder = Path(root_folder)


config = Config()
