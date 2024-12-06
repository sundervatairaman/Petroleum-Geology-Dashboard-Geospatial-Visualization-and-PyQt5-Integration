import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QApplication, QMainWindow, QAction, QVBoxLayout, QWidget, QTextEdit, QComboBox, QPushButton, QLabel, QHBoxLayout, QMdiArea, QMdiSubWindow
from odbind.survey import Survey
from odbind.seismic3d import Seismic3D
from odbind.well import Well
from odbind.horizon3d import Horizon3D
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import xarray as xr
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from PyQt5.QtCore import Qt
from matplotlib.figure import Figure
import pandas as pd
import numpy as np







class SeismicInlineCanvas(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(12, 8))
        super().__init__(fig)
        self.setParent(parent)
        self.survey = Survey('F3_Demo_2020')
        self.seismic = Seismic3D(self.survey, '4 Dip steered median filter')
        self.plot_image()

    def plot_image(self):
        vol = self.seismic.volume[:, :, :]
        firstcomp = self.seismic.comp_names[0]
        sismic_std = np.nanstd(vol[firstcomp], dtype=np.double)
        self.ax.clear()
        xr.plot.imshow(vol[firstcomp].sel(iline=425), ax=self.ax, x='xline', y='twt', yincrease=False, cmap='RdGy', robust=True)
        self.ax.set_title("Seismic Inline at iline 425")
        self.draw()

class PlotWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = plt.Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)
        self.canvas.mpl_connect('scroll_event', self.on_scroll)
        self.canvas.mpl_connect('button_press_event', self.on_button_press)
        self.canvas.mpl_connect('button_release_event', self.on_button_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_mouse_move)
        self.press = None

    def log_plot(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(log_data, depth, label='Log Data')
        if picks is not None:
            ax.scatter(picks, depth, color='red', label='Picks', marker='o')
        ax.set_xlabel(log_name)
        ax.set_ylabel('Depth')
        ax.invert_yaxis()  # Depth increases downward
        ax.legend()
        self.canvas.draw()

    def plot_image(self, image_path):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = plt.imread(image_path)
        height, width, _ = img.shape
        ax.imshow(img, extent=[0, width, 0, height], aspect='auto')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        #ax.invert_yaxis()
        #ax.axis('off')  # Hide the axes 
        #ax.set_aspect('auto')
        self.canvas.draw()

    def plot_vshplot1(self, image_path):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = plt.imread(image_path)
        height, width, _ = img.shape
        ax.imshow(img, extent=[0, width, 0, height], aspect='auto')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        #ax.invert_yaxis()
        #ax.axis('off')  # Hide the axes 
        #ax.set_aspect('auto')
        self.canvas.draw()


    def plot_fig1(self, image_path):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = plt.imread(image_path)
        height, width, _ = img.shape
        ax.imshow(img, extent=[0, width, 0, height], aspect='auto')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        #ax.invert_yaxis()
        #ax.axis('off')  # Hide the axes 
        #ax.set_aspect('auto')
        self.canvas.draw()


    def plot_phit_buck(self, image_path):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = plt.imread(image_path)
        height, width, _ = img.shape
        ax.imshow(img, extent=[0, width, 0, height], aspect='auto')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        #ax.invert_yaxis()
        #ax.axis('off')  # Hide the axes 
        #ax.set_aspect('auto')
        self.canvas.draw()

    def plot_interpetation(self, image_path):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        img = plt.imread(image_path)
        height, width, _ = img.shape
        ax.imshow(img, extent=[0, width, 0, height], aspect='auto')
        ax.set_xlim(0, width)
        ax.set_ylim(0, height)
        #ax.invert_yaxis()
        #ax.axis('off')  # Hide the axes 
        #ax.set_aspect('auto')
        self.canvas.draw()




    def on_scroll(self, event):
        scale_factor = 1.2
        ax = self.figure.gca()
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2

        if event.button == 'up':
            ax.set_ylim([y_center - (y_center - ylim[0]) / scale_factor, y_center + (ylim[1] - y_center) / scale_factor])
        elif event.button == 'down':
            ax.set_ylim([y_center - (y_center - ylim[0]) * scale_factor, y_center + (ylim[1] - y_center) * scale_factor])
        
        self.canvas.draw()

    def on_button_press(self, event):
        if event.button == 1:  # Left mouse button
            self.press = (event.xdata, event.ydata)

    def on_button_release(self, event):
        self.press = None

    def on_mouse_move(self, event):
        if self.press is not None:
            ax = self.figure.gca()
            dx = event.xdata - self.press[0]
            dy = event.ydata - self.press[1]
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            ax.set_xlim([xlim[0] - dx, xlim[1] - dx])
            ax.set_ylim([ylim[0] - dy, ylim[1] - dy])
            self.canvas.draw()


class WellMapWidget(FigureCanvas):
    def __init__(self, parent=None):
        fig, self.ax = plt.subplots(figsize=(12, 8))
        super().__init__(fig)
        self.setParent(parent)
        self.survey = Survey('cambay')
        self.seismic_volume = self.load_seismic_volume()
        self.time_slice = 1200.00
        self.plot()

    def load_seismic_volume(self):
        seismic_volumes = Seismic3D.names(self.survey)
        return Seismic3D(self.survey, seismic_volumes[0])  # Load the first seismic volume

    def update_plot(self, time_slice):
        self.time_slice = time_slice
        self.plot()

    def plot(self):
        # Define the time slice you want to extract
        time_slice = self.time_slice

        # Get the inline and crossline ranges
        inl_range = self.seismic_volume.ranges.inlrg
        crl_range = self.seismic_volume.ranges.crlrg

        # Convert the time slice to a z index (if necessary)
        z_range = [time_slice, time_slice, 1]  # Start, stop, step

        # Fetch the data using getdata
        try:
            data = self.seismic_volume.getdata(inl_range, crl_range, z_range)
        except Exception as e:
            print(f"Error fetching data: {e}")
            return

        # Check the type of the returned result
        if isinstance(data, xr.Dataset):
            data_array = data.to_array().squeeze()  # Convert to DataArray and remove singleton dimensions
            info = {
                'x': data.coords['x'].values,
                'y': data.coords['y'].values
            }
        else:
            raise ValueError("Unexpected return format from getdata.")

        # Extract X and Y coordinates from the info dictionary
        x_coords = np.array(info['x'])
        y_coords = np.array(info['y'])

        # Clear the previous plot
        self.ax.clear()

        # Directly plot using x_coords and y_coords
        contour = self.ax.contourf(x_coords, y_coords, data_array, cmap='seismic')
        self.ax.figure.colorbar(contour, ax=self.ax, label="Amplitude")
        self.ax.set_title(f"Seismic Time Slice at {time_slice} ms")
        self.ax.set_xlabel("X Coordinate")
        self.ax.set_ylabel("Y Coordinate")

        # Get well names
        well_names = Well.names(self.survey)

        # Initialize a dataframe to hold well coordinates
        well_data = []

        # Function to get well coordinates
        def get_well_coordinates(well_name):
            well = Well(self.survey, well_name)
            track_data = well.track()
            x_coords = track_data['x']
            y_coords = track_data['y']
            return x_coords, y_coords

        # Fetch coordinates for each well
        for well_name in well_names:
            x_coords, y_coords = get_well_coordinates(well_name)
            well_data.append({'well_name': well_name, 'x': x_coords, 'y': y_coords})

        # Convert well data to DataFrame
        well_df = pd.DataFrame(well_data)

        # Plot wells
        for index, row in well_df.iterrows():
            x_coords = row['x']
            y_coords = row['y']

            if isinstance(x_coords, np.ndarray) and isinstance(y_coords, np.ndarray):
                for x, y in zip(x_coords, y_coords):
                    self.ax.scatter(x, y, color='red', s=1)
                # Get the last x and y coordinates
                last_x = x_coords[-1]
                last_y = y_coords[-1]
                # Plot the last point and the text label
                self.ax.scatter(last_x, last_y, color='black')
                self.ax.text(last_x, last_y, row['well_name'], fontsize=8, ha='center')
            else:
                # If x and y are single values, plot them directly
                self.ax.scatter(x_coords, y_coords, color='black')
                self.ax.text(x_coords, y_coords, row['well_name'], fontsize=8, ha='center')

        # Set the aspect ratio to be equal
        self.ax.set_aspect('equal')
        self.ax.grid(True)
        self.draw()  # Make sure to call draw to update the plot






class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Survey Information')
        self.setGeometry(100, 100, 1200, 800)

        

        

        


        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)

        # Survey selection components
        self.survey_combo_box = QComboBox()
        self.load_surveys()

        self.load_survey_button = QPushButton('Load Survey')
        self.load_survey_button.clicked.connect(self.load_selected_survey)

        self.survey_label = QLabel('Select Survey:')

        # Well selection components
        self.well_combo_box = QComboBox()
        self.load_well_button = QPushButton('Load Well')
        self.load_well_button.clicked.connect(self.load_selected_well)
        self.well_label = QLabel('Select Well:')

        # Seismic volume selection components
        self.seismic_combo_box = QComboBox()
        self.load_seismic_button = QPushButton('Load Seismic')
        self.load_seismic_button.clicked.connect(self.load_selected_seismic)
        self.seismic_label = QLabel('Select Seismic Volume:')

        # Horizon selection components
        self.horizon_combo_box = QComboBox()
        self.load_horizon_button = QPushButton('Load Horizon')
        self.load_horizon_button.clicked.connect(self.load_selected_horizon)
        self.horizon_label = QLabel('Select Horizon:')

        # Log selection components
        self.log_combo_box = QComboBox()
        self.load_log_button = QPushButton('Load Log')
        self.load_log_button.clicked.connect(self.load_selected_log)
        self.log_label = QLabel('Select Log:')

        # Plot selection components
        self.plot_selection_combo_box = QComboBox()
        self.plot_selection_combo_box.addItems(['Well Map', 'Log Plot','Inline Plot', 'Combo Plot', 'Vshale','Lithology','Rw','Log interpetation' ])
        self.plot_selection_combo_box.currentIndexChanged.connect(self.update_plot_view)
        self.plot_selection_label = QLabel('Select Plot Type:')

        self.f3demo_survey = None  # To store the selected survey
        self.specific_well = None  # To store the selected well
        self.seismic_volume = None  # To store the selected seismic volume

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()

        survey_layout = QVBoxLayout()
        survey_layout.addWidget(self.survey_label)
        survey_layout.addWidget(self.survey_combo_box)
        survey_layout.addWidget(self.load_survey_button)

        well_layout = QVBoxLayout()
        well_layout.addWidget(self.well_label)
        well_layout.addWidget(self.well_combo_box)
        well_layout.addWidget(self.load_well_button)

        seismic_layout = QVBoxLayout()
        seismic_layout.addWidget(self.seismic_label)
        seismic_layout.addWidget(self.seismic_combo_box)
        seismic_layout.addWidget(self.load_seismic_button)

        horizon_layout = QVBoxLayout()
        horizon_layout.addWidget(self.horizon_label)
        horizon_layout.addWidget(self.horizon_combo_box)
        horizon_layout.addWidget(self.load_horizon_button)

        log_layout = QVBoxLayout()
        log_layout.addWidget(self.log_label)
        log_layout.addWidget(self.log_combo_box)
        log_layout.addWidget(self.load_log_button)

        plot_selection_layout = QVBoxLayout()
        plot_selection_layout.addWidget(self.plot_selection_label)
        plot_selection_layout.addWidget(self.plot_selection_combo_box)

        left_layout = QVBoxLayout()
        left_layout.addLayout(survey_layout)
        left_layout.addLayout(well_layout)
        left_layout.addLayout(seismic_layout)
        left_layout.addLayout(horizon_layout)
        left_layout.addLayout(log_layout)
        left_layout.addLayout(plot_selection_layout)
        left_layout.addWidget(self.text_edit)

        self.mdi_area = QMdiArea()

        self.log_plot_window = QMdiSubWindow()
        self.log_plot_widget = PlotWidget()
        self.log_plot_window.setWidget(self.log_plot_widget)
        #self.log_plot_window.setWidget(self.plot_log_widget)

        self.log_plot_window.setWindowTitle("Log Plot")
        self.mdi_area.addSubWindow(self.log_plot_window)

        self.well_map_window = QMdiSubWindow()
        self.well_map_widget = WellMapWidget()
        self.well_map_window.setWidget(self.well_map_widget)
        self.well_map_window.setWindowTitle("Well Map")
        self.mdi_area.addSubWindow(self.well_map_window)

        self.survey_info_window = QMdiSubWindow()
        self.survey_info_widget = QTextEdit()
        self.survey_info_widget.setReadOnly(True)
        self.survey_info_window.setWidget(self.survey_info_widget)
        self.survey_info_window.setWindowTitle("Survey Information")
        self.mdi_area.addSubWindow(self.survey_info_window)

        self.log_plot_window.show()
        self.well_map_window.show()
        self.survey_info_window.show()

        main_layout.addLayout(left_layout)
        main_layout.addWidget(self.mdi_area)

        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

    def load_surveys(self):
        try:
            surveys = Survey.names()
            self.survey_combo_box.addItems(surveys)
        except Exception as e:
            self.text_edit.append(f"Error loading surveys: {e}")

    def load_selected_survey(self):
        selected_survey_name = self.survey_combo_box.currentText()
        try:
            self.f3demo_survey = Survey(selected_survey_name)
            self.text_edit.append(f"Survey '{selected_survey_name}' loaded successfully.")
            self.update_well_combo_box()
            self.update_seismic_combo_box()
            self.update_horizon_combo_box()
        except Exception as e:
            self.text_edit.append(f"Error loading survey: {e}")

    def update_well_combo_box(self):
        self.well_combo_box.clear()
        try:
            wells = Well.names(self.f3demo_survey)
            self.well_combo_box.addItems(wells)
        except Exception as e:
            self.text_edit.append(f"Error updating well combo box: {e}")

    def update_seismic_combo_box(self):
        self.seismic_combo_box.clear()
        try:
            seismic_volumes = Seismic3D.names(self.f3demo_survey)
            self.seismic_combo_box.addItems(seismic_volumes)
        except Exception as e:
            self.text_edit.append(f"Error updating seismic combo box: {e}")

    def update_horizon_combo_box(self):
        self.horizon_combo_box.clear()
        try:
            horizons = Horizon3D.names(self.f3demo_survey)
            self.horizon_combo_box.addItems(horizons)
        except Exception as e:
            self.text_edit.append(f"Error updating horizon combo box: {e}")

    def update_log_combo_box(self):
        self.log_combo_box.clear()
        try:
            log_names = self.specific_well.log_names
            self.log_combo_box.addItems(log_names)
        except Exception as e:
            self.text_edit.append(f"Error updating log combo box: {e}")


    def load_selected_well(self):
        selected_well_name = self.well_combo_box.currentText()
        try:
            self.specific_well = Well(self.f3demo_survey, selected_well_name)
            self.text_edit.append(f"Well '{selected_well_name}' loaded successfully.")
            self.update_log_combo_box()
        except Exception as e:
            self.text_edit.append(f"Error loading well: {e}")

    def load_selected_seismic(self):
        selected_seismic_name = self.seismic_combo_box.currentText()
        try:
            self.seismic_volume = Seismic3D(self.f3demo_survey,selected_seismic_name)
            self.text_edit.append(f"Seismic volume '{selected_seismic_name}' loaded successfully.")
        except Exception as e:
            self.text_edit.append(f"Error loading seismic volume: {e}")

    def load_selected_horizon(self):
        selected_horizon_name = self.horizon_combo_box.currentText()
        try:
            horizon = Horizon3D(self.f3demo_survey,selected_horizon_name)
            self.text_edit.append(f"Horizon '{selected_horizon_name}' loaded successfully.")
        except Exception as e:
            self.text_edit.append(f"Error loading horizon: {e}")

    def load_selected_log(self):
     selected_log_name = self.log_combo_box.currentText()
     try:
        # Retrieve log data
        log_data1, uom = self.specific_well.logs_dataframe([selected_log_name], zstep=0.2, upscale=False)
        log_depth = log_data1['dah'].values
        log_values = log_data1[selected_log_name].values
        
        # Retrieve picks
        well_horizon_picks = self.specific_well.marker_info_dataframe()
        pick_names = well_horizon_picks['name'].values
        pick_colors = well_horizon_picks['color'].values
        pick_depths = well_horizon_picks['dah'].values
        
        # Plot log data and picks
        self.plot_widget.plot_log(log_depth, log_values, selected_log_name)
        
        # Plot horizon picks as horizontal lines and add labels
        ax = self.plot_widget.figure.gca()  # Get current axis
        for pick_depth, pick_color, pick_name in zip(pick_depths, pick_colors, pick_names):
            if pick_depth != -9999:  # Ignore invalid picks
                ax.axhline(y=pick_depth, color=pick_color, linestyle='--', label=pick_name)
                
                # Add pick name as a label
                ax.text(0.05, pick_depth, pick_name, color=pick_color, fontsize=8, verticalalignment='center', 
                        transform=ax.get_yaxis_transform())
        
        # Redraw the canvas to update the plot
        self.plot_widget.canvas.draw()
        self.text_edit.append(f"Log '{selected_log_name}' with picks plotted successfully.")
     except Exception as e:
        self.text_edit.append(f"Error loading log: {e}")



    

    def list_all_seismic(self):
        self.text_edit.clear()
        try:
            seismic_volumes = Seismic3D.names(self.f3demo_survey)
            self.text_edit.append("All Seismic Volumes:")
            for volume in seismic_volumes:
                self.text_edit.append(volume)
        except Exception as e:
            self.text_edit.append(f"Error listing seismic volumes: {e}")

    def list_all_wells(self):
        self.text_edit.clear()
        try:
            wells = Well.names(self.f3demo_survey)
            self.text_edit.append("All Wells:")
            for well in wells:
                self.text_edit.append(well)
        except Exception as e:
            self.text_edit.append(f"Error listing wells: {e}")

    def list_all_logs(self):
        self.text_edit.clear()
        try:
            log_names = self.specific_well.log_names
            self.text_edit.append("Log Names:")
            for log_name in log_names:
                self.text_edit.append(log_name)
        except Exception as e:
            self.text_edit.append(f"Error listing logs: {e}")


    def get_log_info(self):
        self.text_edit.clear()
        try:
            log_info = self.specific_well.log_info
            self.text_edit.append("Log Info:")
            for info in log_info:
                self.text_edit.append(f"{info}: {log_info[info]}")
        except Exception as e:
            self.text_edit.append(f"Error getting log info: {e}")

    def get_marker_info(self):
        self.text_edit.clear()
        try:
            marker_info = self.specific_well.marker_info
            self.text_edit.append("Marker Info:")
            for marker in marker_info:
                self.text_edit.append(f"{marker}: {marker_info[marker]}")
        except Exception as e:
            self.text_edit.append(f"Error getting marker info: {e}")

    def get_track_name(self):
        self.text_edit.clear()
        try:
            track_name = self.specific_well.track_name
            self.text_edit.append(f"Track Name: {track_name}")
        except Exception as e:
            self.text_edit.append(f"Error getting track name: {e}")

    def list_all_horizons(self):
        self.text_edit.clear()
        try:
            horizons = Horizon3D.names(self.f3demo_survey)
            self.text_edit.append("All Horizons:")
            for horizon in horizons:
                self.text_edit.append(horizon)
        except Exception as e:
            self.text_edit.append(f"Error listing horizons: {e}")

    def update_plot_view(self):
        plot_type = self.plot_selection_combo_box.currentText()
        if plot_type == 'Well Map':
            self.plot_well_map()
        elif plot_type == 'Log Plot':
            self.plot_log_data()
        elif plot_type == 'Inline Plot':
            self.plot_inline_data()
        elif plot_type == 'Combo Plot':
            self.plot_image_data()
        elif plot_type == 'Vshale':
            self.plot_vshale_data()
        elif plot_type == 'Lithology':
            self.plot_lith_data()
        elif plot_type == 'Log interpetation':
            self.plot_int_data()

    

    def plot_well_map(self):
        well_map_sub_window = QMdiSubWindow()
        well_map_widget = WellMapWidget()
        well_map_sub_window.setWidget(well_map_widget)
        well_map_sub_window.setWindowTitle("Well Map")
        self.mdi_area.addSubWindow(well_map_sub_window)
        well_map_sub_window.show()

        
        well_map_widget.plot()

    def plot_log_data(self):
        log_sub_window = QMdiSubWindow()
        log_widget = PlotWidget()
        log_sub_window.setWidget(log_widget)
        log_sub_window.setWindowTitle("Log Plot")
        self.mdi_area.addSubWindow(log_sub_window)
        log_sub_window.show()

        
        log_widget.log_plot()

    def plot_inline_data(self):
        image_sub_window = QMdiSubWindow()
        image_widget = SeismicInlineCanvas()
        image_sub_window.setWidget(image_widget)
        image_sub_window.setWindowTitle("Image Plot")
        self.mdi_area.addSubWindow(image_sub_window)
        image_sub_window.show()

        # Example image data
        image_path = 'abcd.png'
        image_widget.plot_image()
        

    def plot_image_data(self):
        image_sub_window = QMdiSubWindow()
        image_widget = PlotWidget()
        image_sub_window.setWidget(image_widget)
        image_sub_window.setWindowTitle("Image Plot")
        self.mdi_area.addSubWindow(image_sub_window)
        image_sub_window.show()

        # Example image data
        image_path = 'triple_combo_plot.png'
        image_widget.plot_image(image_path)
        
        image_sub_window.resize(image_widget.sizeHint())






    def plot_vshale_data(self):
        image_sub_window = QMdiSubWindow()
        image_widget = PlotWidget()
        image_sub_window.setWidget(image_widget)
        image_sub_window.setWindowTitle("Image Plot")
        self.mdi_area.addSubWindow(image_sub_window)
        image_sub_window.show()

        # Example image data
        image_path = 'vsh_plot1.png'
        image_widget.plot_vshplot1(image_path)
        
        image_sub_window.resize(image_widget.sizeHint())

    def plot_lith_data(self):
        image_sub_window = QMdiSubWindow()
        image_widget = PlotWidget()
        image_sub_window.setWidget(image_widget)
        image_sub_window.setWindowTitle("Image Plot")
        self.mdi_area.addSubWindow(image_sub_window)
        image_sub_window.show()

        # Example image data
        image_path = 'fig1.png'
        image_widget.plot_fig1(image_path)
        
        image_sub_window.resize(image_widget.sizeHint())

    def plot_int_data(self):
        image_sub_window = QMdiSubWindow()
        image_widget = PlotWidget()
        image_sub_window.setWidget(image_widget)
        image_sub_window.setWindowTitle("Image Plot")
        self.mdi_area.addSubWindow(image_sub_window)
        image_sub_window.show()

        # Example image data
        image_path = 'interpretation_plot.png'
        image_widget.plot_interpetation(image_path)
        
        image_sub_window.resize(image_widget.sizeHint())





if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec_())
