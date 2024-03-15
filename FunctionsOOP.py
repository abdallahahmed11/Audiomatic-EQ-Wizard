import os
import numpy as np
from PyQt5.QtWidgets import QFileDialog, QMessageBox
import pyqtgraph as pg
# from matplotlib_inline.backend_inline import FigureCanvas
from scipy.fftpack import ifft
from scipy.signal import windows
import scipy.signal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import matplotlib.pyplot as plt
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer
from scipy.io import wavfile
from scipy.io import wavfile
# from pydub import AudioSegment
import sounddevice as sd



class SignalProcessor:
    def __init__(self, main_app):
        self.main_app = main_app
        self.signal = None
        self.fft = None
        self.new_fft_result = None
        self.window_type = 'Rectangle'
        self.sample_rate = None
        self.player = None
        self.current_sample = 0



    def load_signal(self , graph):
        filepath, _ = QFileDialog.getOpenFileName(self.main_app, "Open File", "", "Data Files (*.dat *.csv);;Sound Files (*.wav)")
        if filepath:
            _, extension = os.path.splitext(filepath)
            if not os.path.exists(filepath):
                QMessageBox.critical(self.main_app, "File Not Found", f"Could not find file at {filepath}.")
                return
            data = None
            if extension == '.dat':
                # Read the .dat file as 16-bit integers
                data = np.fromfile(filepath, dtype=np.int16)
                time = data[:,0]
                self.sample_rate = 1.0 / (time[1] - time[0])
            elif extension == '.csv':
                # data = np.loadtxt(filepath, delimiter=',', skiprows=1 ,max_rows=10000)
                data = np.loadtxt(filepath, delimiter=',', skiprows=1 ,max_rows=10000 , usecols=(0,1))
                time = data[:,0]
                self.sample_rate = 1.0 / (time[1] - time[0])

            elif extension == '.wav':
                sampling_rate, audio_data = wavfile.read(filepath)
                if audio_data.ndim > 1:
                    audio_data = audio_data[:, 0]

                audio_data = audio_data.astype(np.float32) / np.iinfo(audio_data.dtype).max
                time = np.arange(0, len(audio_data)) / sampling_rate
                data = np.column_stack((time, audio_data))
                self.player = QMediaPlayer()
                media_content = QMediaContent(QUrl.fromLocalFile(filepath))
                self.player.setMedia(media_content)
                self.sample_rate = sampling_rate
                if self.player.state() == QMediaPlayer.StoppedState:
                    self.player.play()

            self.signal = data  # Assign the loaded data to self.signal
            self.plot_signal(data ,graph)
            self.get_freq_components(data )
            self.main_app.inputGraph1.getViewBox().autoRange()
            self.main_app.inputGraph2.getViewBox().autoRange()
            self.main_app.inputGraph3.getViewBox().autoRange()
            self.main_app.inputGraph4.getViewBox().autoRange()




    def plot_signal(self, data , graph):
        graph.clear()
        graph.addItem(pg.PlotDataItem(data))

    def dynamic_plotting(self, data, graph):
        current_sample = 0
        graph.clear()

        # Create a plot item in the graph
        self.curve = graph.plot()

        # Update data
        self.curve.setData(data[:current_sample])
        graph.setXRange(max(0, current_sample - 100), current_sample)

        # Set graph limits
        graph.setLimits(xMin=0, xMax=current_sample + 100, yMin=0, yMax=1.1)

        # Update the current sample
        current_sample += 1

        if current_sample >= len(data):
            self.timer.stop()

        # Add a grid
        graph.showGrid(x=True, y=True)

        # Continue the timer
        self.timer.start(1000)

    def rfft(self, signal, n_samples):
        # Compute the positive-frequency terms with rfft and scale it
        return np.fft.rfft(signal) / n_samples

    def get_mag_and_phase(self, fft_result):
        # Compute the magnitudes and phases
        magnitudes = np.abs(fft_result) * 2
        phases = np.angle(fft_result)
        return magnitudes, phases

    def change_magnitude(self, magnitude, window):
        new_magnitude = magnitude * window
        return new_magnitude

    def create_equalized_signal(self, magnitudes, phases):
        # Create a new fft result with modified magnitudes and original phases
        new_fft_result = magnitudes * np.exp(1j * phases)
        return new_fft_result

    def inverse(self, new_fft_result, n_samples):
        return np.fft.irfft(new_fft_result * n_samples)  # Scale by len(sig) because of the earlier scaling

    def get_freq(self, n_samples, sampling_rate):
        # Compute the frequency bins
        frequencies = np.fft.rfftfreq(n_samples, sampling_rate)
        return frequencies

    def apply_windowing(self, range ,slider_value , window_type):
        # window = scipy.signal.windows.boxcar(len(signal))
        # windowed_signal = signal * window
        # return windowed_signal
        if window_type == 'Rectangle':
            window = scipy.signal.windows.boxcar(range,slider_value)
        elif window_type == 'Hamming':
            window = scipy.signal.windows.hamming(range,slider_value)
        elif window_type == 'Hanning':
            window = scipy.signal.windows.hann(range,slider_value)
        elif window_type == 'Gaussian':
            window = scipy.signal.windows.gaussian(range,slider_value)

        windowed_signal = window * slider_value

        return windowed_signal





    def get_freq_components(self, signal):
        # get time and Amplitude
        time = signal[:, 0]
        Amplitude = signal[:, 1]
        sampling_rate = 1.0 / (time[1] - time[0])
        n_samples = len(Amplitude)

        # Compute the Fast Fourier Transform (FFT)
        self.fft = self.rfft(Amplitude, n_samples)

        # Compute the frequencies associated with the FFT
        freqs = self.get_freq(n_samples, 1.0 / sampling_rate)

        # Find the corresponding magnitudes of the positive frequencies
        magnitude, phases = self.get_mag_and_phase(self.fft)

        # Create 10 equal frequency ranges
        freq_boundaries = np.linspace(0, max(freqs), 10)
        freq_ranges = []
        for i in range(1, len(freq_boundaries)):
            freq_range = (freq_boundaries[i - 1], freq_boundaries[i])
            freq_ranges.append(freq_range)

        return freq_ranges, magnitude, phases, freqs, time




    def plot_equalized_fft(self, equalized_sig, sampling_rate , freqGraph):
        n_samples = len(equalized_sig)

        # Compute the FFT of the equalized signal
        equalized_fft = self.rfft(equalized_sig, n_samples)

        # Compute the frequencies associated with the FFT
        freqs = self.get_freq(n_samples, sampling_rate)


        freqGraph.clear()
        freqGraph.addItem(pg.PlotDataItem(freqs, np.abs(equalized_fft) * 2, pen='r') ) # Plot the magnitude spectrum
        # freqGraph.addItem(pg.PlotDataItem(self.apply_windowing(self.main_app.windowSlider4.value(), 1, self.window_type)))

        freqGraph.setLabel('left', 'Magnitude')
        freqGraph.setLabel('bottom', 'Frequency')

    def clear_modes(self, modes,clear_spectrogram=True):
        # Define all the graph views for each mode
        graph_views = {
            1: ['inputGraph1', 'outputGraph1', 'freqGraph1', 'windowGraph1'],
            2: ['inputGraph2', 'outputGraph2', 'freqGraph2', 'windowGraph2'],
            3: ['inputGraph3', 'outputGraph3', 'freqGraph3', 'windowGraph3'],
            4: ['inputGraph4', 'outputGraph4', 'freqGraph4', 'windowGraph4'],
        }

        # Iterate over the modes and clear the corresponding graph views
        for mode in modes:
            for view in graph_views[mode]:
                getattr(self.main_app, view).clear()

        # If clear_spectrogram is True, clear the spectrogram
        if clear_spectrogram:
            if hasattr(self, 'spectrogram_plotter'):
                self.spectrogram_plotter.clear_spectrogram()
            if hasattr(self, 'spectrogram_plotter_2'):
                self.spectrogram_plotter_2.clear_spectrogram()


    def on_window_type_changed3(self, index, comboBox, sliderValue):
        # Determine the active tab
        active_tab_index = self.main_app.tabWidget.currentIndex()
        self.sliderValue= sliderValue
        # Get the selected window type from the ComboBox
        window_type_index = comboBox.currentIndex()
        window_types = ['Rectangle', 'Hamming', 'Hanning', 'Gaussian']
        selected_window_type = window_types[window_type_index] if window_type_index < len(window_types) else 'Rectangle'

        # Mapping of tab index to corresponding graph widget
        window_graph_mapping = {
            0: self.main_app.windowGraph1,
            1: self.main_app.windowGraph2,
            2: self.main_app.windowGraph3,
            3: self.main_app.windowGraph4,
            # 0: self.main_app.freqGraph1,
            # 1: self.main_app.freqGraph2,
            # 2: self.main_app.freqGraph3,
            # 3: self.main_app.freqGraph4
        }

        # Get the corresponding graph for the current tab
        graph = window_graph_mapping.get(active_tab_index)
        if graph is not None:
            self.window_type = selected_window_type
            graph.clear()
            graph.addItem(pg.PlotDataItem(self.apply_windowing(sliderValue, 1, self.window_type)))

        else:
            # Handle case where tab index is not in the mapping (if necessary)
            self.window_type = selected_window_type






    def default_graph_drawing_2(self, sliderValue):
        # Mapping of tab index to corresponding graph
        window_graph_mapping = {
            0: self.main_app.windowGraph1,
            1: self.main_app.windowGraph2,
            2: self.main_app.windowGraph3,
            3: self.main_app.windowGraph4
        }

        active_tab_index = self.main_app.tabWidget.currentIndex()

        # Get the corresponding graph for the current tab
        graph = window_graph_mapping.get(active_tab_index)
        if graph is not None:
            graph.clear()
            graph.addItem(pg.PlotDataItem(self.apply_windowing(sliderValue, 1, 'Rectangle')))



    def zoomOut(self, graph1 , graph2):
        # You can adjust the zoom factor as needed
        zoom_factor = 1.2
        graph1.getViewBox().scaleBy((zoom_factor, zoom_factor))
        graph2.getViewBox().scaleBy((zoom_factor, zoom_factor))

    def zoomIn(self, graph1 , graph2):
        # You can adjust the zoom factor as needed
        zoom_factor = 0.8
        graph1.getViewBox().scaleBy((zoom_factor, zoom_factor))
        graph2.getViewBox().scaleBy((zoom_factor, zoom_factor))

    def fitScreen(self, graph1 , graph2):
        graph1.getViewBox().autoRange()
        graph2.getViewBox().autoRange()



    def get_freq_components_sound(self, signal, index):
        time, amplitude = signal[:, 0], signal[:, 1]
        n_samples = len(amplitude)

        self.fft = self.rfft(amplitude, n_samples)
        freqs = self.get_freq(n_samples, 1.0 / self.sample_rate)
        magnitude, phases = self.get_mag_and_phase(self.fft)

        freq_ranges = self.calculate_freq_ranges(index, max(freqs))

        return freq_ranges, magnitude, phases, freqs, time

    def calculate_freq_ranges(self, index, max_freq):
        freq_ranges_dict = {
            0: [(0, 500), (500, 1000), (1000, 2000), (2000, 20000)],
            1: [(0, 1100), (1100, 2500), (4200, 6000), (7000, 22000)],
            2: [(1, 200), (1 / 18, 1 / 9), (27.7777, 55.55), (0.0, 12)],
            3: self.create_linear_freq_ranges(max_freq, 10)
        }

        return freq_ranges_dict.get(index, "Invalid index for frequency ranges.")

    def create_linear_freq_ranges(self, max_freq, num_ranges):
        freq_boundaries = np.linspace(0, max_freq, num_ranges + 1)
        return [(freq_boundaries[i], freq_boundaries[i + 1]) for i in range(num_ranges)]



    def apply_equalizer_sound(self, freq_ranges, magnitude, phases, freqs, time, freqGraph, outputTimeGraph, loopIterations, sliderName):
        for i in range(loopIterations):
            slider_value = getattr(self.main_app, f'{sliderName}{i+1}').value()
            idx = np.where((freqs >= freq_ranges[i][0]) & (freqs < freq_ranges[i][1]))
            window = self.apply_windowing(len(idx[0]), slider_value, self.window_type)
            magnitude[idx] = self.change_magnitude(magnitude[idx], window)

        self.new_fft_result = self.create_equalized_signal(magnitude, phases)
        equalized_sig = self.inverse(self.new_fft_result, len(self.fft))
        outputTimeGraph.clear()
        min_length = min(len(time), len(equalized_sig))
        time = time[:min_length]
        equalized_sig = equalized_sig[:min_length]
        outputTimeGraph.addItem(pg.PlotDataItem(time, equalized_sig))
        self.plot_equalized_fft(equalized_sig, 1.0 / self.sample_rate, freqGraph)
        if sliderName in ["instrumentSlider_", "animalSlider_"]:
            self.playProcessedSound(equalized_sig)
        return equalized_sig




    def playProcessedSound(self,processedData):
        sd.play(processedData,self.sample_rate)

    def play_pause_audio(self):
        if hasattr(self, 'player') and self.player:
            if self.player.state() == QMediaPlayer.PlayingState:
                self.player.pause()
            else:
                self.player.play()





class SpectrogramPlotter:
    def __init__(self,layout):
        self.figure = None
        self.axes = None
        self.Spectrogram = None
        self.layout = layout
        self.signal_processor = SignalProcessor(self)

    def create_spectrogram_figure_and_plot(self,data,sample_rate):
        """Create a new spectrogram figure with black background."""
        # If a canvas already exists, remove it from the layout
        if self.Spectrogram is not None:
            self.layout.removeWidget(self.Spectrogram)
            self.Spectrogram.deleteLater()
            self.Spectrogram = None


        # Create new matplotlib Figure and Axes
        self.figure, self.axes = plt.subplots()
        self.figure.patch.set_facecolor('white')

        # Plot the spectrogram on the axes
        self.axes.specgram(data, Fs=sample_rate, cmap='viridis',
                           aspect='auto')

        # Create canvas to be displayed in the layout
        self.Spectrogram = Canvas(self.figure)

        # Add the Spectrogram canvas to the passed layout
        self.layout.addWidget(self.Spectrogram)

        # add axis





    def clear_spectrogram(self):
        """Clear the current spectrogram."""
        self.layout.removeWidget(self.Spectrogram)
        self.Spectrogram.deleteLater()
        self.Spectrogram = None



