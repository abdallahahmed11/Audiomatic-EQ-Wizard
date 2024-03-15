from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout
from PyQt5.uic import loadUiType
import os
from os import path
import sys
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
# import the function from utils.py
from FunctionsOOP import SignalProcessor, SpectrogramPlotter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
import scipy.signal
# import UI file
FORM_CLASS, _ = loadUiType(path.join(path.dirname(__file__), "Equalizer-All.ui"))


class MainApp(QMainWindow, FORM_CLASS):
    def __init__(self, parent=None):
        super(MainApp, self).__init__(parent)
        self.setupUi(self)

        # Create an instance of SignalProcessor
        self.signal_processor = SignalProcessor(self)

        # Add your custom logic and functionality here
        self.Handle_Buttons()
        self.intializer()



    def intializer(self):
        self.tabWidget.setCurrentIndex(0)
        self.window_type='Rectangle'
        self.uniformSlider_1.setRange(0, 10)
        self.uniformSlider_2.setRange(0, 10)
        self.uniformSlider_3.setRange(0, 10)
        self.uniformSlider_4.setRange(0, 10)
        self.uniformSlider_5.setRange(0, 10)
        self.uniformSlider_6.setRange(0, 10)
        self.uniformSlider_7.setRange(0, 10)
        self.uniformSlider_8.setRange(0, 10)
        self.uniformSlider_9.setRange(0, 10)
        self.uniformSlider_10.setRange(0, 10)
        self.uniformSlider_1.setValue(1)
        self.uniformSlider_2.setValue(1)
        self.uniformSlider_3.setValue(1)
        self.uniformSlider_4.setValue(1)
        self.uniformSlider_5.setValue(1)
        self.uniformSlider_6.setValue(1)
        self.uniformSlider_7.setValue(1)
        self.uniformSlider_8.setValue(1)
        self.uniformSlider_9.setValue(1)
        self.uniformSlider_10.setValue(1)



        self.windowSlider1.setRange(100, 1000)
        self.windowSlider2.setRange(100, 1000)
        self.windowSlider3.setRange(100, 1000)
        self.windowSlider4.setRange(100, 1000)



        self.ecgSlider_1.setValue(1)
        self.ecgSlider_2.setValue(1)
        self.ecgSlider_3.setValue(1)
        self.ecgSlider_4.setValue(1)
        self.ecgSlider_1.setRange(0,2)
        self.ecgSlider_2.setRange(0,2)
        self.ecgSlider_3.setRange(0,2)
        self.ecgSlider_4.setRange(0,2)

        self.instrumentSlider_1.setRange(0,2)
        self.instrumentSlider_2.setRange(0,2)
        self.instrumentSlider_3.setRange(0,2)
        self.instrumentSlider_4.setRange(0,2)

        self.instrumentSlider_1.setValue(1)
        self.instrumentSlider_2.setValue(1)
        self.instrumentSlider_3.setValue(1)
        self.instrumentSlider_4.setValue(1)


        self.animalSlider_1.setRange(0,2)
        self.animalSlider_2.setRange(0,2)
        self.animalSlider_3.setRange(0,2)
        self.animalSlider_4.setRange(0,2)
        self.animalSlider_1.setValue(1)
        self.animalSlider_2.setValue(1)
        self.animalSlider_3.setValue(1)
        self.animalSlider_4.setValue(1)

        # self.timer = QTimer()
        # self.timer.timeout.connect(self.signal_processor.update_plot)




    def Handle_Buttons(self):
        self.tabWidget.currentChanged.connect(self.tab_changed_handler)

    def apply_equalizer(self, freqGraph, outputTimeGraph, mode_index):
        # Map mode_index to the corresponding parameters for get_freq_components_sound and slider prefix
        mode_settings = {
            4: (3,"uniformSlider_",9),
            2: (1, "animalSlider_",4),
            1: (0, "instrumentSlider_",4),
            3: (2, "ecgSlider_",4)
        }

        # Fetch the settings based on mode_index
        component_index, slider_prefix,loopIterations = mode_settings.get(mode_index, (None, None))

        # If component_index is None, the mode_index is invalid; handle this case appropriately
        if component_index is None:
            raise ValueError(f"Invalid mode_index: {mode_index}")

        # Get frequency components based on the mode_index
        freq_ranges, magnitude, phases, freqs, time = self.signal_processor.get_freq_components_sound(
            self.signal_processor.signal, component_index)

        self.spectrogram_plotter.create_spectrogram_figure_and_plot(time, self.signal_processor.sample_rate)

        # Apply equalizer based on sound


        time_of_equalized_signal=self.signal_processor.apply_equalizer_sound(freq_ranges, magnitude, phases, freqs, time, freqGraph,
                                                    outputTimeGraph, loopIterations, slider_prefix)
        self.spectrogram_plotter_2.create_spectrogram_figure_and_plot(time_of_equalized_signal, self.signal_processor.sample_rate)




    def tab_changed_handler(self, index):
        if index == 0:
            print("First tab clicked")
            # self.browseButton1.clicked.connect(self.signal_processor.load_signal)
            self.browseButton1.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.inputGraph1))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_16)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_17)
            self.browseButton1.clicked.connect(
                lambda: self.signal_processor.default_graph_drawing_2(sliderValue=self.windowSlider1.value()))            # self.applyButton1.clicked.connect(self.apply_equalizer_handler)
            self.applyButton1.clicked.connect(lambda:self.apply_equalizer(freqGraph=self.freqGraph1 , outputTimeGraph=self.outputGraph1,mode_index=4))
            self.signal_processor.clear_modes([2, 3, 4])
            self.comboBox_mode1.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed3(index=0, comboBox=self.comboBox_mode1,
                                                                      sliderValue=self.windowSlider1.value()))
            self.zoomInButton1.clicked.connect(lambda :self.signal_processor.zoomIn(graph1=self.inputGraph1 , graph2=self.outputGraph1))
            self.zoomOutButton1.clicked.connect(lambda :self.signal_processor.zoomOut(graph1=self.inputGraph1, graph2=self.outputGraph1))
            self.fitButton1.clicked.connect(lambda :self.signal_processor.fitScreen(graph1=self.inputGraph1 ,  graph2=self.outputGraph1))




        elif index == 1:
            print("Second tab clicked")
            # self.browseButton2.clicked.connect(self.signal_processor.load_signal)
            self.browseButton2.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.inputGraph2))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_22)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_23)
            # self.browseButton2.clicked.connect(self.signal_processor.default_graph_drawing)
            # self.browseButton2.clicked.connect(
            self.browseButton2.clicked.connect(
                lambda: self.signal_processor.default_graph_drawing_2(sliderValue=self.windowSlider2.value()))
            # self.applyButton2.clicked.connect(lambda:self.apply_sound_equalizer_handler(freqGraph=self.freqGraph2, outputTimeGraph= self.outputGraph2))
            self.applyButton2.clicked.connect(lambda:self.apply_equalizer(freqGraph=self.freqGraph2, outputTimeGraph= self.outputGraph2 , mode_index=1))

            self.signal_processor.clear_modes([1, 3, 4])
            self.comboBox_mode2.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed3(index=0, comboBox=self.comboBox_mode2,
                                                                      sliderValue=self.windowSlider2.value()))
            self.zoomInButton2.clicked.connect(lambda: self.signal_processor.zoomIn(graph1=self.inputGraph2 ,  graph2=self.outputGraph2))
            self.zoomOutButton2.clicked.connect(lambda: self.signal_processor.zoomOut(graph1=self.inputGraph2 ,  graph2=self.outputGraph2))
            self.fitButton2.clicked.connect(lambda: self.signal_processor.fitScreen(graph1=self.inputGraph2 ,  graph2=self.outputGraph2))
            self.playButton2.clicked.connect(lambda: self.signal_processor.play_pause_audio())


        elif index == 2:
            print("Third tab clicked")
            # self.browseButton3.clicked.connect(self.signal_processor.load_signal)
            self.browseButton3.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.inputGraph3))
            self.browseButton3.clicked.connect(
                lambda: self.signal_processor.default_graph_drawing_2(sliderValue=self.windowSlider3.value()))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_20)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_21)

            self.applyButton3.clicked.connect(lambda:self.apply_equalizer(freqGraph=self.freqGraph3, outputTimeGraph= self.outputGraph3 , mode_index=2))

            self.signal_processor.clear_modes([1, 2, 4])
            self.comboBox_mode3.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed3(index=0, comboBox=self.comboBox_mode3,
                                                                      sliderValue=self.windowSlider3.value()))

            self.zoomInButton3.clicked.connect(lambda: self.signal_processor.zoomIn(graph1=self.inputGraph3 ,graph2=self.outputGraph3))
            self.zoomOutButton3.clicked.connect(lambda: self.signal_processor.zoomOut(graph1=self.inputGraph3 , graph2=self.outputGraph3))
            self.fitButton3.clicked.connect(lambda: self.signal_processor.fitScreen(graph1=self.inputGraph3,  graph2=self.outputGraph3))
            self.playButton3.clicked.connect(lambda: self.signal_processor.play_pause_audio())



        elif index == 3:
            print("Fourth tab clicked")
            self.browseButton4.clicked.connect(lambda: self.signal_processor.load_signal(graph=self.inputGraph4))
            self.browseButton4.clicked.connect(
                lambda: self.signal_processor.default_graph_drawing_2(sliderValue=self.windowSlider4.value()))
            self.spectrogram_plotter = SpectrogramPlotter(self.verticalLayout_18)
            self.spectrogram_plotter_2 = SpectrogramPlotter(self.verticalLayout_19)

            self.applyButton4.clicked.connect(lambda:self.apply_equalizer(freqGraph=self.freqGraph4, outputTimeGraph= self.outputGraph4 , mode_index=3))

            self.signal_processor.clear_modes([1, 2, 3])
            self.comboBox_mode4.currentIndexChanged.connect(
                lambda: self.signal_processor.on_window_type_changed3(index=0, comboBox=self.comboBox_mode4,
                                                                      sliderValue=self.windowSlider4.value()))

            self.zoomInButton4.clicked.connect(lambda: self.signal_processor.zoomIn(graph1=self.inputGraph4 ,  graph2=self.outputGraph4))
            self.zoomOutButton4.clicked.connect(lambda: self.signal_processor.zoomOut(graph1=self.inputGraph4 ,  graph2=self.outputGraph4))
            self.fitButton4.clicked.connect(lambda: self.signal_processor.fitScreen(graph1=self.inputGraph4 ,  graph2=self.outputGraph4))




def main():
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()