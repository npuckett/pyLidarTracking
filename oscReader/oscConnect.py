import sys
import math
import numpy as np
import threading
from sklearn.cluster import DBSCAN
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore

class LiDARPlotter:
    def __init__(self, plot_widget, point_size=5, x_offset=0, y_offset=0, rotation_offset=0):
        self.scatter = pg.ScatterPlotItem(size=point_size, pen=pg.mkPen(None))
        self.cluster_scatter = pg.ScatterPlotItem(size=point_size, pen=None)  # No pen for cluster points
        plot_widget.addItem(self.scatter)
        plot_widget.addItem(self.cluster_scatter)
        plot_widget.setXRange(-10000, 10000)  # Adjust as needed
        plot_widget.setYRange(-10000, 10000)  # Adjust as needed
        self.current_buffer = []
        self.last_angle = None
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.rotation_offset = rotation_offset
        self.eps = 1  # DBSCAN distance parameter
        self.min_samples = 5  # Minimum samples for DBSCAN

    def add_point(self, angle, distance):
        if self.last_angle is not None and angle < self.last_angle:
            self.update_plot()
            self.current_buffer = []

        # Apply offset and rotation to the point
        rotated_angle = angle + self.rotation_offset
        x = (distance * math.cos(math.radians(rotated_angle))) + self.x_offset
        y = (distance * math.sin(math.radians(rotated_angle))) + self.y_offset
        self.current_buffer.append((x, y))
        self.last_angle = angle

    def update_plot(self):
        if not self.current_buffer:
            return

        # Convert polar to Cartesian coordinates
        x, y = zip(*self.current_buffer)
        self.scatter.setData(x, y)

        # Perform DBSCAN clustering
        points = np.column_stack((x, y))
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit(points)
        labels = clustering.labels_

        # Count and print the number of clusters (excluding noise)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"End of rotation: {len(self.current_buffer)} points, {n_clusters} clusters.")

        # Clear previous cluster scatter plots
        self.cluster_scatter.clear()

        # Plot clusters with different colors
        unique_labels = set(labels)
        colors = [pg.intColor(i, hues=len(unique_labels)) for i in range(len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:  # Noise
                continue
            class_member_mask = (labels == k)
            xy_cluster = points[class_member_mask]
            self.cluster_scatter.addPoints(xy_cluster[:, 0], xy_cluster[:, 1], brush=col)

    def set_eps(self, value):
        self.eps = value

    def set_min_samples(self, value):
        self.min_samples = value

def handle_lidar(address, *args):
    if address == "/lidar1":
        angle, distance, point_number = args
        plotter.add_point(angle, distance)

dispatcher = Dispatcher()
dispatcher.map("/lidar1", handle_lidar)

def start_osc_server():
    server = BlockingOSCUDPServer(("127.0.0.1", 8000), dispatcher)
    print("Starting OSC server")
    server.serve_forever()

def create_slider_with_label(label_text, min_val, max_val, step, initial, callback):
    layout = QtWidgets.QHBoxLayout()
    label = QtWidgets.QLabel(label_text)
    value_label = QtWidgets.QLabel(str(initial))
    slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
    slider.setRange(min_val, max_val)
    slider.setSingleStep(step)
    slider.setValue(initial)
    slider.valueChanged.connect(lambda value: [callback(value), value_label.setText(str(value))])
    layout.addWidget(label)
    layout.addWidget(slider)
    layout.addWidget(value_label)
    return layout


# PyQtGraph setup
app = QtWidgets.QApplication(sys.argv)
win = QtWidgets.QWidget()
layout = QtWidgets.QVBoxLayout()
plot_widget = pg.GraphicsLayoutWidget()
plot = plot_widget.addPlot(title="LiDAR Data")
plotter = LiDARPlotter(plot, point_size=5, x_offset=0, y_offset=0, rotation_offset=0)

# Add the plot widget to the layout
layout.addWidget(plot_widget)

# Add sliders
eps_slider_layout = create_slider_with_label("EPS:", 1, 100, 1, 1, lambda value: plotter.set_eps(value))
min_samples_slider_layout = create_slider_with_label("Min Samples:", 1, 10, 1, 5, lambda value: plotter.set_min_samples(value))
layout.addLayout(eps_slider_layout)
layout.addLayout(min_samples_slider_layout)

# Set layout and show window
win.setLayout(layout)
win.show()

# Start OSC server in a separate thread
thread = threading.Thread(target=start_osc_server)
thread.start()

# Start Qt event loop
sys.exit(app.exec_())
