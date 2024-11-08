import requests
import zipfile
import os
from tqdm import tqdm


# function to download the zipfile data
def download_zip(url, path_output):

    # download zip directory develop
    try:
        # Send a request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the ZIP file temporarily
        zip_file_path = "{}.zip".format(path_output.split("/")[-1])
        with open(zip_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Extract the contents of the ZIP file
        unzip(path_zip_file=zip_file_path, path_output=path_output)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download ZIP file: {e}")
    except zipfile.BadZipFile:
        print(f"Failed to extract: Not a valid ZIP file")


# unzip data
def unzip(path_zip_file, path_output):
    try:
        # Open the ZIP file
        with zipfile.ZipFile(path_zip_file, "r") as zip_ref:
            # Extract all files to the specified directory
            zip_ref.extractall(path_output)
            print(
                f"ZIP file '{path_zip_file}' extracted successfully to '{path_output}'"
            )

        # Remove the temporary ZIP file
        os.remove(path_zip_file)

    except zipfile.BadZipFile:
        print(f"Error: '{path_zip_file}' is not a valid ZIP file")


# download data function
def download_data_unzip(data_name, url=None, data_machine=None):

    # data directory path
    path_data_directory = os.path.dirname(os.path.abspath(__file__))

    # data path
    path_data = os.path.join(path_data_directory, data_name)

    # check if exist
    machine_output_path = [os.path.join(path_data, m) for m in data_machine]
    check_machine = [os.path.exists(mp) for mp in machine_output_path]

    # downlaod the zip
    if not os.path.exists(path_data) or not all(check_machine):
        download_zip(url=url, path_output=path_data)
        path_machine_zip = [os.path.join(path_data, i) for i in os.listdir(path_data)]

        # unzip each machine
        for mp_zip in tqdm(
            path_machine_zip,
            total=len(path_machine_zip),
            desc="Unzipping {} machine files".format(data_name),
        ):
            unzip(path_zip_file=mp_zip, path_output=path_data)

    else:
        print(f"{path_data} is already downloaded")
        for mp in machine_output_path:
            print(f"{mp} is already downloaded")


# develop dataset
develop_name = "develop"
develop_url = "https://seafile.cloud.uni-hannover.de/f/451c8bf1f0074684a442/?dl=1"
develop_machine = ["bearing", "fan", "gearbox", "slider", "ToyCar", "ToyTrain", "valve"]
download_data_unzip(
    data_name=develop_name,
    data_machine=develop_machine,
    url=develop_url,
)
