import os
import shutil
import tarfile
import zipfile
from typing import List, Tuple
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from PIL import Image


def find_or_create_dir(dir_name: str) -> str:
    """Creates a directory if it doesn't exist.

    Args:
        dir_name (str): The name of the directory to create if it doesn't
        exist.

    Returns:
        The local path of the directory.
    """
    # Get the directory of the current file.
    parent_dir_path = os.path.dirname(os.path.realpath(__file__))
    
    # Create a directory if it doesn't exist.
    dir_path = os.path.join(parent_dir_path, dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return dir_path


def download_remote_data_file(local_data_folder: str, 
    data_url: str) -> Tuple[str, bool]:
    """Downloads data from url if it's not saved locally yet.

    Args:
        data_url (str): The url of the data file we want to download.
    
    Returns:
        The path to the local file, and a bool indicating whether the file
        was downloaded or not.
    """
    # Create a data directory if it doesn't exist.
    data_dir_path = find_or_create_dir(local_data_folder)
    
    # Download the data file if it doesn't exist.
    filename = os.path.basename(urlparse(data_url).path)
    data_file_path = os.path.join(data_dir_path, filename)
    downloaded = False
    if not os.path.exists(data_file_path):
        print(f'Downloading data file {data_file_path}...')
        with urlopen(data_url) as response:
            with open(data_file_path, "wb") as data_file:
                shutil.copyfileobj(response, data_file)
        downloaded = True
        print('Done downloading data file.')

    return (data_file_path, downloaded)


def unpack_zip_file(folder: str, zip_filename: str) -> None:
    """Unpacks a zip file into the same folder.

    Args:
        folder (str): The folder where the zip file is contained, and where it
        will get unpacked.

        zip_filename (str): The name of the zip file.
    """    
    local_file_zip = os.path.join(folder, zip_filename)
    with zipfile.ZipFile(local_file_zip) as myzip:
        myzip.extractall(folder)


def unpack_tar_file(folder: str, tar_filename: str) -> None:
    """Unpacks a tar file into the same folder.

    Args:
        folder (str): The folder where the tar file is contained, and where it
        will get unpacked.

        tar_filename (str): The name of the tar file.
    """    
    local_file_tar = os.path.join(folder, tar_filename)
    with tarfile.TarFile(local_file_tar) as myzip:
        myzip.extractall(folder)


def append_to_all_files(folder: str, to_append: str) -> None:
    """Appends a string to the end of all filenames in the specified folder, 
    if they don't already end with that string.

    Args:
        folder (str): The folder where the files to be renamed are located.

        to_append (str): String to append to all filenames.
    """
    filenames = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for filename in filenames:
        if not filename.endswith(to_append):
            new_filename = filename + to_append
            os.rename(os.path.join(folder, filename), os.path.join(folder, new_filename))


def convert_images(folder: str, out_ext: str) -> None:
    """Converts all images in folder specified from their current extension
    to the specified one and saves them. Works recursively.

    Args:
        folder (str): Folder where images are located and will be saved to.

        out_ext (str): The new extension we want images to have.
    """
    contents = os.listdir(folder)
    for content in contents:
        content = os.path.join(folder, content)
        if os.path.isdir(content):
            convert_images(content, out_ext)
        elif os.path.isfile(content):
            in_file = content
            base, ext = os.path.splitext(in_file)
            out_file = base + out_ext
            if in_file != out_file:
                try:
                    Image.open(in_file).save(out_file)
                except IOError:
                    print(f'Cannot convert file {in_file} to {out_file}.')
