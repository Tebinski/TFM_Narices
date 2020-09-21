import os

def get_list_of_files(folder):
    """Returns a list with the files contained in that folder"""
    ls_files = os.listdir(folder)
    path_files = [os.path.join(folder, f) for f in ls_files]
    return path_files

def get_list_of_files_with_extension(folder, ext):
    """Returns a list with the files with the specified extension {ext} contained in that folder."""
    path_files = get_list_of_files(folder)
    ls_ext = [f for f in path_files if f.endswith(ext)]
    return ls_ext

