import requests
import zipfile
import os
from tqdm import tqdm


# function to download the zipfile data
def download_zip(url, output_path):

    # download zip directory develop
    try:
        # Send a request to the URL
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Check if the request was successful

        # Save the ZIP file temporarily
        zip_file_path = "{}.zip".format(develop_name)
        with open(zip_file_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        # Extract the contents of the ZIP file
        unzip(zip_file_path=zip_file_path, output_path=output_path)

    except requests.exceptions.RequestException as e:
        print(f"Failed to download ZIP file: {e}")
    except zipfile.BadZipFile:
        print(f"Failed to extract: Not a valid ZIP file")


# unzip data
def unzip(zip_file_path, output_path):
    try:
        # Open the ZIP file
        with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
            # Extract all files to the specified directory
            zip_ref.extractall(output_path)
            print(
                f"ZIP file '{zip_file_path}' extracted successfully to '{output_path}'"
            )

        # Remove the temporary ZIP file
        os.remove(zip_file_path)

    except zipfile.BadZipFile:
        print(f"Error: '{zip_file_path}' is not a valid ZIP file")


# download data function
def download_data_unzip(data_name, url=None, data_machine=None):

    # data directory path
    data_directory_path = os.path.dirname(os.path.abspath(__file__))

    # data path
    data_path = os.path.join(data_directory_path, data_name)

    # check if exist
    machine_output_path = [os.path.join(data_path, m) for m in data_machine]
    check_machine = [os.path.exists(mp) for mp in machine_output_path]

    # downlaod the zip
    if not os.path.exists(data_path) or not all(check_machine):
        download_zip(url=url, output_path=data_path)
        machine_zip_path = [os.path.join(data_path, i) for i in os.listdir(data_path)]

        # unzip each machine
        for mp_zip in tqdm(
            machine_zip_path,
            total=len(machine_zip_path),
            desc="Unzipping {} machine files".format(data_name),
        ):
            unzip(zip_file_path=mp_zip, output_path=data_path)

    else:
        print(f"{data_path} is already downloaded")
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
