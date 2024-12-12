## Instructions on how to run the project

1. Download the required files from the repo & [shared Google Drive folder](https://drive.google.com/open?id=1B7x_Fxfa2GDd9qmNs73by3O6N8uZEC_9&usp=drive_fs) (for dataset). Minimally required files to run the streamlit app are:
    * `streamlit.py`: Main source files
    * `requirements.txt`: pip requirements file
    * `processed_data.pickle`: The bulk of the required data pre-computed for the app. This file serves to reduce computation time & avoid unpredictable caching issues with streamlit.
    
2. Place all downloaded files under the same directory, navigate to the directory in the terminal, create a new virtual environment (tested with `conda` on Python 3.11.10) and run the following command to install the required packages:
    ```
    pip install -r requirements.txt
    ```

3. Launch the streamlit app with the following command:
    ```
    streamlit run streamlit.py
    ```

4. The app will open in a new tab in your default browser. You can interact with the app via the select boxes on the interface.

**Note**:  
During testing we experienced some random crashing of the app, and the traceback seemed to point to some deeply buried issues with streamlit's caching mechanism (via `@st.cache_data`). We therefore removed caching from the app & moved the bulk of the data processing to a pre-computed file (`processed_data.pickle`). We haven't experienced any crashes since then. However, if you experience any issues, please try relaunching the app via the command in step 3. If the issue persists, please let us know.

## Other non-essential files in the repo & shared folder
* `Archive`: Subdirectory that contains older versions of our model & related files
* `capstone.ipynb`: Model training logic & part of the user interface logic implemented with `ipywidgets`
* `data.ipynb`: Data downloading via the [EODHD](https://eodhd.com/) API
* `sp500.csv`: S&P 500 constituents list
* Data files that have been incorporated into `processed_data.pickle`:
  * `autoencoder_vol.keras`: Pre-trained autoencoder model
  * `history_vol.pickle`: Training history of the autoencoder model
  * `val_mae_loss.pickle`: Validation loss plot of the autoencoder model


