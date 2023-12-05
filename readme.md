# Task 2. Computer vision. Sentinel-2 image matching

## Section 1: Project Structure

- `archive`: The data folder that you will prepare in the next section. Dataset with the Sentinel2 satellite images.
- `kp_mathing.py`: Python script (.py) for model inference.
- `kp_mathing.ipynb`: Jupyter Notebook with a detailed solving problem explanation.
- `result_{sample}.jpg`: Examples of the model's performance on various data.


## Section 2: Environment Setup

To run the project, an isolated environment is recommended to manage dependencies and ensure consistency. Here's how to set up the Anaconda environment for this project:

1. **Create a New Conda Environment**:
   - Open the Anaconda Command Prompt.
   - Use the command `conda create --name myenv` (replace `myenv` with your desired environment name).
   - Activate the new environment using `conda activate myenv`.

2. **Install Dependencies**:
   - Navigate to the root project directory.
   - Run `pip install -r requirements.txt` to install the necessary packages.

3. **Download Dataset**:
   - Go to this site: https://www.kaggle.com/datasets/isaienkov/deforestation-in-ukraine/data.
   - Download data from this competition
   - Unzip the data in your project directory

Once the installation is complete, the environment is ready. You can now execute any script from the project within this Conda environment.
