import os


def make_folder(folder, **kwargs):
    """Utility to make folders

    Args:
        folder (string): name of folder

    Returns:
        string: path to folder
    """
    # Makes folder
    os.makedirs(folder, exist_ok=True)

    return folder
