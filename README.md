Geospatial Visualization and PyQt5 Integration
This project integrates Dash for web-based visualizations with PyQt5 for desktop applications, focusing on geospatial data visualization using Plotly and shapefiles. It combines interactive data rendering with a user-friendly interface for enhanced usability.

Features
Load and display various geospatial datasets in shapefile format.
Interactive visualizations using Dash and Plotly.
PyQt5 integration for desktop GUIs with web-based visualization components.
Support for shapefiles like tectonic plates, cratons, oil and gas fields, and geological features.
Handles geopandas-based geospatial data processing.
Prerequisites
Ensure you have the following installed:

Python 3.10 or higher
Required Python libraries:
Dash
Plotly
GeoPandas
PyQt5
Matplotlib
Install dependencies via pip:

bash
Copy code
pip install dash plotly geopandas pyqt5 matplotlib
Setup
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/geospatial-visualization.git
cd geospatial-visualization
Ensure your shapefiles are placed in the assets/ directory, or update the base_path in the script if the directory is different.

Run the script:

bash
Copy code
python main.py
Directory Structure
bash
Copy code
geospatial-visualization/
├── assets/                # Directory for shapefiles and static files
│   ├── plates.shp         # Example shapefile
│   ├── cratons.shp        # Example shapefile
│   └── ...
├── main.py                # Main application script
├── README.md              # Project documentation
└── requirements.txt       # Python dependencies
Usage
Start the Application: Launch the app using python main.py.

Load Shapefiles: The application will automatically load shapefiles from the specified paths. If any shapefiles are missing, an error message will be printed.

Visualize Data:

View interactive Dash plots in your browser.
Use the PyQt5 interface for additional functionality.
Supported Shapefiles
The app supports the following datasets:

Tectonic plates
Cratons
Oil and gas fields
Geological features
Large Igneous Provinces (LIPs)
Troubleshooting
Ensure all shapefile paths are correct and files exist in the specified locations.
If using frozen applications (e.g., PyInstaller), the base_path logic ensures asset paths are resolved correctly.
