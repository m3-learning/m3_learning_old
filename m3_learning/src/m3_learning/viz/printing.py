import matplotlib.pyplot as plt
from ..util.file_IO import make_folder


class printer:
    """Class to save figures to a folder"""

    def __init__(self, dpi=600, basepath="./", fileformats=["png", "svg"]):
        """Initializes the printer class

        Args:
            dpi (int, optional): the resolution of the image. Defaults to 600.
            basepath (str, optional): basepath where files are saved. Defaults to './'.
        """
        self.dpi = dpi
        self.basepath = basepath
        self.fileformats = fileformats
        make_folder(self.basepath)

    def savefig(self, fig, name, tight_layout=False, basepath=None):
        """Function to save a figure

        Args:
            fig (figure): figure to save
            name (str): file name to save the figure
        """
        if tight_layout:
            fig.tight_layout()

        if basepath is None:
            basepath = self.basepath

        for fileformat in self.fileformats:
            print(basepath + name + "." + fileformat)
            fig.savefig(
                basepath + name + "." + fileformat,
                dpi=self.dpi,
                bbox_inches="tight",
            )
