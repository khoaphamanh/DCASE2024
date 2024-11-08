import requests
import zipfile
import os


# function to download the zipfile model
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


# unzip model
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


# download models function
def download_data_unzip(model_name, url=None):

    # models directory path
    models_directory_path = os.path.dirname(os.path.abspath(__file__))

    # prertrained models directory path
    prertrained_models_directory = "pretrained_models"
    path_prertrained_models_directory = os.path.join(
        models_directory_path, prertrained_models_directory
    )
    os.makedirs(path_prertrained_models_directory, exist_ok=True)

    path_prertrained_models_name = os.path.join(
        path_prertrained_models_directory, model_name
    )

    # check the downloaded models
    if not os.path.exists(path_prertrained_models_name):

        # download model as zip file
        download_zip(url=url, path_output=path_prertrained_models_directory)

        # unzip the model
        unzip(
            path_zip_file=path_prertrained_models_directory,
            path_output=path_prertrained_models_name,
        )

    else:
        print(f"{path_prertrained_models_name} is already downloaded")


# model beats
beat_iter_3_url = "https://seafile.cloud.uni-hannover.de/f/aa2e6f0e57094a1ea338/?dl=1"
beat_iter_3_name = "BEATs_iter3.pt"

download_data_unzip(
    model_name=beat_iter_3_name,
    url=beat_iter_3_url,
)
