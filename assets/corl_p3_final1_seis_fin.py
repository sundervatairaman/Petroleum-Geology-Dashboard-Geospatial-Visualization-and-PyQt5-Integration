import os
import sys
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QTextEdit
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import QUrl
from PyQt5.QtWidgets import QListWidget, QLabel, QPushButton, QCheckBox, QGridLayout
import matplotlib.pyplot as plt
import xarray as xr
from odbind.survey import Survey
from odbind.seismic3d import Seismic3D
from odbind.well import Well
from odbind.horizon3d import Horizon3D
import pandas as pd
from PyQt5.QtWidgets import QApplication, QListWidget, QVBoxLayout, QLabel, QPushButton, QWidget, QAbstractItemView, QCheckBox, QGridLayout
from PyQt5.QtWebChannel import QWebChannel
from PyQt5.QtCore import QUrl, QObject, pyqtSlot
from plotly.subplots import make_subplots
from arb18_fin_plt_log import SeismicHorizonPlotter 
from scipy.interpolate import interp1d


import matplotlib.cm as cm
from matplotlib.colors import Normalize


# Define the colormap and normalization
cmap = cm.viridis  # Choose the colormap (you can change this to any other colormap)
norm = Normalize(vmin=0, vmax=150)  # Set the color range from 0 to 150


# Define your Survey and Well classes (assuming these are correct)
survey = Survey('D2')
wells = Well.names(survey)
print(wells)
well_name = 'KJ-6'
base_path = r'C:\open_dtect'
survey_name = 'D2'
well_folder = 'WellInfo'

horizon_names = Horizon3D.names(survey)
inline_number = 797
time_slice = 1250  # Example time slice
file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name}.wlt"
depth_to_time_file_path = file_path


# Define the functions for depth-to-time conversion
def load_depth_to_time_model(file_path):
    depths = []
    times = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() and not line.startswith('!') and not line.startswith('Name') and not line.startswith('Description'):
                try:
                    depth, time = map(float, line.split())
                    depths.append(depth)
                    times.append(time)
                except ValueError:
                    # Handle lines that can't be converted to float (possibly headers or comments)
                    pass
    return np.array(depths), np.array(times)

def create_interpolation_function(depths, times):
    return interp1d(depths, times, kind='linear', fill_value='extrapolate')

def depth_to_time(depth, interp_func):
    return interp_func(depth)




def plot_track_data(well: Well):
    """Plot well track data."""
    try:
        # Retrieve well track data as a DataFrame
        track_data = well.track_dataframe()
        print("Track Data:", track_data)  # Debugging print
        x_coord=track_data['x']
        y_coord=track_data['y']
        print(x_coord[1],y_coord[1])
        if track_data.empty:
            print("No track data available.")
            return

        #plt.figure(figsize=(12, 8))
        #for column in track_data.columns:
            #plt.plot(track_data[column],track_data.index,  label=column)
        
        #plt.xlabel('Depth')
        #plt.ylabel('Value')
        #plt.title('Well Track Data')
        #plt.legend()
        #plt.grid(True)
        #plt.show()
        return x_coord[1],y_coord[1]
    except Exception as e:
        print(f"Error retrieving track data: {e}")



def generate_well_map_figure(selected_wells=None):
    

    if selected_wells is None:
        selected_wells = []

    fig = go.Figure()

    well_coordinates = {}
    for well_no in wells:
        specific_well = Well(survey, well_no)
        trk = specific_well.track()

        x_coords = trk['x']
        y_coords = trk['y']

        last_x = x_coords[-1]
        last_y = y_coords[-1]

        color = 'blue' if well_no in selected_wells else 'red'

        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y],
            mode='markers+text',
            text=[well_no],
            marker=dict(color=color, size=10),
            textposition='top center',
            customdata=[well_no],  # Add customdata to store well numbers
            hoverinfo='text',
            name=well_no
        ))

        well_coordinates[well_no] = (last_x, last_y)

    fig.update_layout(
        #title='Well Map - All Wells',
        xaxis_title='X',xaxis=dict(domain=[0.7, 1],),
        yaxis_title='Y',
        showlegend=False,yaxis=dict(scaleanchor="x", scaleratio=1,domain=[0.6, 1],),
        clickmode='event+select'  # Enable click events
    )

    return fig

def generate_connection_map(selected_wells, well_coordinates,fig):
    #fig = go.Figure()

    well_coordinates = {}
    for well_no in wells:
        specific_well = Well(survey, well_no)
        trk = specific_well.track()

        x_coords = trk['x']
        y_coords = trk['y']

        last_x = x_coords[-1]
        last_y = y_coords[-1]

        color = 'blue' if well_no in selected_wells else 'red'

        fig.add_trace(go.Scatter(
            x=[last_x], y=[last_y],
            mode='markers+text',
            text=[well_no],
            marker=dict(color=color, size=5),
            textposition='top center',textfont=dict(size=10), 
            customdata=[well_no],  # Add customdata to store well numbers
            hoverinfo='text',
            name=well_no,xaxis='x3',yaxis='y3',
        ),row=1,col=20,secondary_y=True,)

        well_coordinates[well_no] = (last_x, last_y)



    # Plot selected wells
    for well_no in selected_wells:
        if well_no in well_coordinates:
            x, y = well_coordinates[well_no]
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode='markers+text',
                text=[well_no],
                marker=dict(color='blue', size=5),
                textposition='top center',textfont=dict(size=10),
                name=well_no,xaxis='x3',yaxis='y3',

            ),row=1,col=20,secondary_y=True,)

    # Connect selected wells with blue lines
    for i in range(len(selected_wells) - 1):
        well_no1 = selected_wells[i]
        well_no2 = selected_wells[i + 1]
        if well_no1 in well_coordinates and well_no2 in well_coordinates:
            x1, y1 = well_coordinates[well_no1]
            x2, y2 = well_coordinates[well_no2]
            fig.add_trace(go.Scatter(
                x=[x1, x2], y=[y1, y2],
                mode='lines',
                line=dict(color='blue', width=2),
                name=f'{well_no1} to {well_no2}',xaxis='x3',yaxis='y3',

            ),row=1,col=20,secondary_y=True,)
    y_values = [coords[1] for coords in well_coordinates.values()]
    x_values = [coords[0] for coords in well_coordinates.values()]

    y_range=max(y_values)-min(y_values)
    fig.update_yaxes(  row=1, col=20,domain=[0.9, 1])
    fig.update_xaxes(  row=1, col=20,domain=[0.9, 1])

    fig.update_layout(
        #title='Well Map with Connections',
        #xaxis3=dict(domain=[0.7, 1],),
        #yaxis3=dict(domain=[0.6, 1],),

        showlegend=False,
        #yaxis=dict(scaleanchor="x", scaleratio=1)
    )

    return fig

class WebChannel(QObject):
    @pyqtSlot(str)
    def select_well(self, well_no):
        print(f"Selected Well: {well_no}")
        self.parent().update_selected_well(well_no)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Well Map and Data Visualization')
        self.setGeometry(100, 100, 1200, 800)  # Set window dimensions

        

        self.html_file = os.path.abspath("corl_plot.html")
        self.connection_html_file = os.path.abspath("connection_plot.html")
        self.corl_html_file = os.path.abspath("corl_plot.html")

        self.selected_wells = []
        

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)

        

        

        # Create a QTextEdit to display logs
        self.log_text_edit = QTextEdit()
        self.log_text_edit.setReadOnly(True)
        #self.layout.addWidget(self.log_text_edit)

        # Initialize the GUI for well map selection and plotting
        self.init_gui()

        

        # Add the QTextEdit (log area) at the bottom
        self.layout.addWidget(self.log_text_edit)  # Add this last to keep it at the bottom
        
        # Set up the web channel
        self.channel = QWebChannel()
        self.web_channel = WebChannel()
        self.web_channel.setParent(self)
        self.channel.registerObject('backend', self.web_channel)
        self.web_view.page().setWebChannel(self.channel)

        # Log information
        self.log("Plotly plot created and saved successfully.")

    def init_gui(self):
      grid_layout = QGridLayout()  # Create a grid layout
      # Create a QWebEngineView and load the Plotly HTML file
      self.web_view = QWebEngineView()
      self.web_view.setUrl(QUrl.fromLocalFile(self.html_file))
      self.web_view.setMinimumHeight(900)  # Adjust the height as needed
      grid_layout.addWidget(self.web_view, 0,40)
      self.plot_well_map()  # Initial plot


      well_nos = wells

      #well_label = QLabel('Select Well:')
      #self.layout.addWidget(well_label)  # Add label to the main layout

      #self.well_list = QListWidget()
      #self.well_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)  # Enable multi-selection
      #self.well_list.addItems(wells)
      #grid_layout.addWidget(self.well_list, 0, 1, 1, 2)  # Add list to grid layout

      

      # Add checkboxes for well selection directly on the map
      self.well_checkboxes = {}  # Dictionary to store checkboxes
      checkbox_layout = QVBoxLayout()
      for i, well_no in enumerate(well_nos):
        checkbox = QCheckBox(well_no)
        checkbox.setChecked(False)  # Default to selected
        checkbox.stateChanged.connect(self.on_checkbox_state_changed)  # Connect to state change handler
        self.well_checkboxes[well_no] = checkbox
        checkbox_layout.addWidget(checkbox)   # 
      
      # Create a QWidget to hold the checkboxes and set the layout
      checkbox_widget = QWidget()
      checkbox_widget.setLayout(checkbox_layout)

      # You would now need to add the checkbox_widget to your overall layout
      # For example, if using QGridLayout for the map and controls:
      grid_layout.addWidget(checkbox_widget, 0, 0)  # Add to the layout next to the map    



      # Create a QListWidget to display selected wells
      self.selected_well_list = QListWidget()
      grid_layout.addWidget(self.selected_well_list, 80, 40, len(well_nos), 1)  # Add to the grid
      
      plot_button = QPushButton('Plot Well Map')
      plot_button.clicked.connect(self.plot_well_map)
      grid_layout.addWidget(plot_button, len(well_nos) + 0, 0)  # Add button to grid

      clear_plot_button = QPushButton('Clear Plot')
      clear_plot_button.clicked.connect(self.clear_plot)
      grid_layout.addWidget(clear_plot_button, len(well_nos) + 1, 0)  # Add button to grid

      print_button = QPushButton('Print Selected Wells')
      print_button.clicked.connect(self.print_selected_wells)
      grid_layout.addWidget(print_button, len(well_nos) + 2, 0)  # Add button to grid

      self.layout.addLayout(grid_layout)  # Add grid layout to the main layout

      



    def on_checkbox_state_changed(self, state):
        # Determine which checkboxes are checked
        selected_wells = [well_no for well_no, checkbox in self.well_checkboxes.items() if checkbox.isChecked()]
        print("Selected wells updated:", selected_wells)
        self.update_selected_well(selected_wells)

    def update_selected_well(self, selected_wells):
        # Clear and update the selected wells list
        self.selected_wells = selected_wells
        self.selected_well_list.clear()
        self.selected_well_list.addItems(self.selected_wells)
        #self.update_connection_map()
        self.print_selected_wells()  # Print the selected wells whenever updated

    #def print_selected_wells(self):
        #print("Selected Wells:")
        #for well in self.selected_wells:
            #print(well)

        # Optionally, also display in the log
        #self.log("Selected Wells:\n" + "\n".join(self.selected_wells))

    
    



    def plot_well_map(self):
      # Plot all wells, with selected ones in blue and unselected ones in red
      fig = generate_well_map_figure()

      # Save the plot as an HTML file
      pio.write_html(fig, file=self.html_file, auto_open=False, include_plotlyjs='cdn')

      # Load the updated plot into the web view
      self.web_view.setUrl(QUrl.fromLocalFile(self.html_file))

      self.log("Well map plot updated.")

    def update_connection_map(self,fig):
      if self.selected_wells:
        well_coordinates = {}
        
        for well_no in self.selected_wells:
            specific_well = Well(survey, well_no)
            track_data = specific_well.track()
            
            # Print track_data to debug
            print(f"Track data for well {well_no}: {track_data}")

            # Adjust this line depending on the structure of track_data
            if isinstance(track_data, dict):
                # Assuming track_data contains 'x' and 'y' keys
                last_x = track_data.get('x', [])[-1] if 'x' in track_data and len(track_data['x']) > 0 else None
                last_y = track_data.get('y', [])[-1] if 'y' in track_data and len(track_data['y']) > 0 else None
            else:
                # If track_data is a DataFrame, use iloc as before
                last_x = track_data.iloc[-1]['x']
                last_y = track_data.iloc[-1]['y']
            
            well_coordinates[well_no] = (last_x, last_y)

        # Generate the connection map
        

        fig = generate_connection_map(self.selected_wells, well_coordinates,fig)
        
        # Overlay the connection plot on top of the existing well map
        # Make sure to use a full map base if necessary, or adjust the update logic

        # Save the connection plot as an HTML file
        #pio.write_html(fig, file=self.corl_html_file, auto_open=False, include_plotlyjs='cdn')

        # Load the connection plot into the web view
        #self.web_view.setUrl(QUrl.fromLocalFile(self.connection_html_file))

        self.log("Connection plot updated.")
        return fig

    def update_map_with_connections(self):
      # Plot all wells
      self.plot_well_map()
    
      # Update the connection map on top of the existing map
      #self.update_connection_map()
#################################################################

# Define a function well corl
    

    def print_selected_wells(self):

     plotter = SeismicHorizonPlotter(survey, 'D2', horizon_names, inline_number, time_slice,self.corl_html_file)
     

     #def plot_well_corl(self):

     alias = {
            'sonic': ['none', 'DTC', 'DT24', 'DTCO', 'DT', 'AC', 'AAC', 'DTHM'],
            'ssonic': ['none', 'DTSM', 'DTS'],
            'GR': ['none', 'GR', 'GRD', 'CGR', 'GRR', 'GRCFM'],
            'RT': ['none', 'HDRS', 'LLD', 'M2RX', 'MLR4C', 'RD', 'RT90', 'RLA1', 'RDEP', 'RLLD', 'RILD', 'ILD', 'RT_HRLT', 'RACELM', 'RT'],
            'resshal': ['none', 'LLS', 'HMRS', 'M2R1', 'RS', 'RFOC', 'ILM', 'RSFL', 'RMED', 'RACEHM'],
            'RHOZ': ['none', 'ZDEN', 'RHOB', 'RHOZ', 'RHO', 'DEN', 'RHO8', 'BDCFM'],
            'NPHI': ['none', 'CNCF', 'NPHI', 'NEU'],
            'pe': ['none','PE', 'PEF', 'PEFZ'],  # Photoelectric factor aliases
            'caliper': ['none','CALI', 'CAL', 'CALI2', 'CALX'],  # Caliper log aliases
            'bs': ['none','BS', 'BIT', 'BITSIZE', 'BDT'],  # Bit size aliases
            'vpvs': ['none','VPVS', 'VP_VS', 'VPVS_RATIO', 'VPVS_R'],  # Vp/Vs ratio aliases
            'rxo': ['none','RXO', 'RMLL', 'Rxo', 'RTXO', 'RSXO', 'RMSFL', 'MSFL', 'RXXO', 'RX', 'M2R6'],
            'sp': ['none','SP', 'SSP', 'SPONT', 'SPONPOT', 'SPOT', 'SPT', 'SP_CURVE']
        }

     

     


     abcd = ['3']
     colors = ['red', 'blue', 'green', 'purple', 'orange', 'pink', 'cyan', 'brown', 'gray', 'magenta',
          'teal', 'lime', 'indigo', 'yellow', 'black', 'lime', 'indigo', 'yellow', 'black']

     #well_names = ['WADU-79','kj3','NDSN-37','WADU-79','WADU-67','WADU-67','WADU-67','WADU-67','WADU-67','WADU-67']  # specify well names
     well_names = self.selected_wells
     log_names = ['GR', 'RT', 'NPHI', 'RHOZ']
     #log_names = ['gr', 'resdeep', 'neutron', 'density']

     min1 = [0, 1, 0.9, 1.65,-1000]
     max1 = [150, 20, 0.1, 2.65,1000]
     horizon_picks = [  'KIIIA-SILT', 'KIII-COAL2', 'KIIIB-SILT', 'KIV-COAL1', 'KIV-SILT', 'KIV_sand_base', 'KV-COAL', 'KV-COAL-BASE', 'KVII_SAND', 'KVIII-COAL', 'KIX-COAL', 'KIX_COAL_BASE', 'KX-COAL', 'KXSAND', 'KXI', 'YCS']
     logs_data = []
     depth_min = float('100')
     depth_max = float('1800')
     log_data = []

   
  


     global_depth_min = float('inf')
     global_depth_max = float('-inf')

    # Step 1: Determine the global depth range
     for well_name in well_names:
          file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name}.wlt"
          
          depths, times = load_depth_to_time_model(file_path)
          interp_func = create_interpolation_function(depths, times)
          well = Well(survey, well_name)
          log_data1, uom = well.logs_dataframe(log_names, zstep=0.1, upscale=False)
        
          #log_depth = log_data1['dah'].values


          depths_from_well = log_data1['dah']
          times = [depth_to_time(depth, interp_func) for depth in depths_from_well]
          log_data1['twt']=times
          md_values = log_data1['dah']
          tvdss_values = [well.tvdss(md) for md in md_values]  # Call the method for each MD value
          log_data1['tvdss'] = tvdss_values
          log_depth = log_data1['twt']*1000



          global_depth_min = min(global_depth_min, np.min(log_depth))
          global_depth_max = max(global_depth_max, np.max(log_depth))

    # Define a common depth track
     common_depth = np.linspace(global_depth_min, global_depth_max, num=8000)

    # Step 2: Interpolate log data to the common depth track
     for well_name in well_names:
          file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name}.wlt"
          
          depths, times = load_depth_to_time_model(file_path)
          interp_func = create_interpolation_function(depths, times)


          well = Well(survey, well_name)
          log_data1, uom = well.logs_dataframe(log_names, zstep=0.1, upscale=False)

          #log_depth = log_data1['dah'].values

          depths_from_well = log_data1['dah']
          times = [depth_to_time(depth, interp_func) for depth in depths_from_well]
          log_data1['twt']=times
          md_values = log_data1['dah']
          tvdss_values = [well.tvdss(md) for md in md_values]  # Call the method for each MD value
          log_data1['tvdss'] = tvdss_values
          log_depth = log_data1['twt']*1000



          logs = []

          for log_name in log_names:
            alias[log_name] = [elem for elem in log_data1 if elem in set(alias[log_name])]
            print(alias[log_name][0])
            print(log_name)
            #log_data1[log_name]=log_data1[alias[log_name][0]].values
            log_data = log_data1[alias[log_name][0]].values
            if isinstance(log_data, (np.ndarray, pd.Series)):
                # Interpolate log data to the common depth
                log_data_interp = np.interp(common_depth, log_depth, log_data)
                logs.append(log_data_interp)
            else:
                print(f"Unexpected log data format for '{log_name}'.")
        
          logs_data.append(logs)

     #return common_depth, logs_data

     log_depth = common_depth
     print("Log Data test:", logs_data[0][0])

     #data['GR']= data[alias['gr'][0]].values
     #if not alias['sonic']:
       # data['DT'] = 0
    
     #else:
        #data['DT'] = data[alias['sonic'][0]].values


     #fig, axs = plt.subplots(len(abcd), (len(well_names)+4)*len(log_names), figsize=(28, 8), sharey='row', sharex='col' )
     #self.update_connection_map()

    ##################################
     # Define the number of subplots (rows = len(well_names), cols = len(log_names))
     rows = 1
     #cols = (len(well_names)+4)*len(log_names)
     cols = 20
     # Create subplots
     fig = make_subplots(
        rows=rows, cols=cols,
        shared_yaxes=True, shared_xaxes=False,
        subplot_titles=log_names,specs=[[{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},
                                         {"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},
                                         {"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},
                                         {"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True},{"secondary_y": True}]],


        horizontal_spacing=0.001
     )

     for i, well_name1 in enumerate(well_names):
        well_name = Well(survey, well_name1)
        wellno = well_name
        well_horizon_picks = well_name.marker_info_dataframe()
        well1 = Well(survey, well_name1)

        

        file_path = f"{base_path}\\{survey_name}\\{well_folder}\\{well_name1}.wlt"
          
        depths, times = load_depth_to_time_model(file_path)
        interp_func = create_interpolation_function(depths, times)


          

          



        pick = well_horizon_picks['dah']
        col = well_horizon_picks['color']
        sand = well_horizon_picks['name']

        depths_from_well = well_horizon_picks['dah']

        times = [depth_to_time(depth, interp_func) for depth in depths_from_well]
        well_horizon_picks['twt']=times
        pick = well_horizon_picks['twt']*1000



        x_coord,y_coord=plot_track_data(well1)
        if i == 0:
            start_coords1 = (x_coord+100,y_coord+100)
        end_coords1 = (x_coord,y_coord)
        if i > 0:
            start_coords1 = (x_coord1,y_coord1)
        
        x_coord1=x_coord
        y_coord1=y_coord
        




        for j, log_name in enumerate(log_names):
            
            log_data = logs_data[i][j]
            if j == 2:
               log_datan = logs_data[i][j + 1]
               x_data_filtered1 = log_datan
               y_data_filtered1 = common_depth
               non_nan_mask1 = ~np.isnan(x_data_filtered1)
 
            if j == 0:
                log_data_1 = np.array(log_data[1])
            
                x_data_filtered2 = log_data


            # Ensure log_data has the right shape and dimensions
            if log_data.ndim == 1:
              x_data = log_data
              y_data = common_depth
            elif log_data.ndim == 2:
              x_data = log_data[0]
              y_data = log_data[1]
            else:
              print(f"Unexpected log data shape for '{log_name}'.")
              continue

            # Plot with x_data as the log values and y_data as depth
            non_nan_mask = ~np.isnan(x_data)
            


            # Add main log data plot
            if j < 4:
            
             fig.add_trace(
                go.Scatter(
                    x=x_data[non_nan_mask],  # Align x values with non-NaN y values
 
                    y=y_data[non_nan_mask], 
                    mode='lines',
                    line=dict(color=colors[j], width=0.5),
                    name=log_name
                ),
                row=1, col=j+(i*5) + 2
             )
             #ax = axs[(j)+(i*5)]


            # Add secondary log (twin axis)
            if j == 0:
                 plotter.plot_arbitrary_seismic_linexy_horizon(start_coords1, end_coords1,j+(i*5) + 1,fig)
  
            if j == 6:   

                fig.add_trace(
                    go.Scatter(
                        x=x_data_filtered1[non_nan_mask1], 
                        y=y_data_filtered1[non_nan_mask1],
                        mode='lines',
                        line=dict(color='red', width=0.5),
                        xaxis=f'x{cols+1}',  # Overlay on a secondary x-axis
                        name=f'{log_names[j + 1]} (Twin)'
                    ),
                    row=1, col=j+(i*5) + 2
                )
                
                # Set x-axis for twin axis
                fig.update_xaxes(range=[min1[j+1], max1[j+1]], row=1, col=j + 2)

            
            
            # Set axes properties
            if j != 1:
                 fig.update_xaxes(range=[min1[j], max1[j]], row= 1, col=j+(i*5) + 2)
            fig.update_yaxes(range=[depth_max, depth_min], autorange="reversed", row=1, col=j+(i*5) + 2)
            fig.update_xaxes(title_text=log_name, row=1, col=j+(i*5) + 2)

            # Customize x-axis tick labels
            if j == 1:
                fig.update_xaxes(type='log', row=1, col=j+(i*5) + 2)  # Log scale

            # Plot horizon picks as horizontal lines
            for k, horizon_pick in enumerate(pick):
                if horizon_pick != -9999:
                    fig.add_trace(
                        go.Scatter(
                            x=[min1[j], max1[j]], y=[horizon_pick, horizon_pick],
                            mode='lines',
                            line=dict(color='blue', width=0.5),
                            name=sand[k]
                        ),
                        row=1, col=j+(i*5) + 2
                    )
                    if j == 3:
                        fig.add_annotation(
                            x=max1[j] - 0.5,
                            y=horizon_pick,
                            text=sand[k],
                            showarrow=False,
                            font=dict(size=6, color='red'),
                            row=1, col=j+(i*5) + 2
                        )
            #fig.update_xaxes(range=[min1[j], max1[j]], row= 1, col=j+(i*5) + 1)



     # Customize layout
     fig.update_layout(
        height=2000, width=1800,
        title_text="Well Log Correlation Plot",
        legend_title_text="Logs",
        showlegend=True,
        
     )

     # Add overall depth labels
     for depth in range(int(depth_min), int(depth_max) + 1, 100):
        fig.add_annotation(
            x=-10, y=depth,
            text=str(depth),
            showarrow=False,
            font=dict(size=6),
            

        )
     
     # Update the connection map on top of the existing map
     fig=self.update_connection_map(fig)

     #fig.show()
     #plotter.plot_arbitrary_seismic_linexy_horizon(start_coords1, end_coords1,j+(i*5) + 2)
     #Save the connection plot as an HTML file
     pio.write_html(fig, file=self.corl_html_file, auto_open=False, include_plotlyjs='cdn')

     # Load the connection plot into the web view
     self.web_view.setUrl(QUrl.fromLocalFile(self.corl_html_file))

     self.log("corl plot updated.")







##############################################################
    def clear_plot(self):
        self.selected_wells.clear()
        self.selected_well_list.clear()
        self.plot_well_map()  # Refresh main map

    def log(self, message):
        # Append messages to the QTextEdit
        self.log_text_edit.append(message)


# Run the application
app = QApplication(sys.argv)
window = MainWindow()
window.show()
sys.exit(app.exec_())
