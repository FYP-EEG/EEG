"""
Author: Anson Li
Created on: 12/6/2026
Purpose: script to receive data from headset
Location: project_dir/EEG/data_receive.py
"""

import argparse
import datetime as dt
from pathlib import Path

import cleaning as cl

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import numpy as np

from pylsl import StreamInfo, StreamOutlet
#py data_receive.py --board-id -2 --ip-address 225.1.1.1 --ip-port 6677 --master-board -1

board = None
args = None
outlet = None
domain = "t"

def config(BID=-2, port=4):
    """
    Edit by Anson
    Date: 18/7/2026
    Changes: added parameters for board id and port for data_record
    """
    global board, args
    BoardShim.enable_dev_board_logger()

    parser = argparse.ArgumentParser()
    # use docs to check which parameters are required for specific board, e.g. for Cyton - set serial port
    parser.add_argument('--timeout', type=int, help='timeout for device discovery or connection', required=False,
                        default=0)
    parser.add_argument('--file', type=str, help='file', required=False, default='')
    #default to UDP
    parser.add_argument('--ip-protocol', type=int, help='ip protocol, check IpProtocolType enum', required=False,
                        default=0)
    #for cyton usb, keep this for wired headset + cyton
    #windows com3/com4...
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default=f"COM{port}")
    #bluetooth for ganglion
    parser.add_argument('--mac-address', type=str, help='mac address', required=False, default='')
    parser.add_argument('--other-info', type=str, help='other info', required=False, default='')
    parser.add_argument('--serial-number', type=str, help='serial number', required=False, default='')
    
    parser.add_argument('--ip-port', type=int, help='ip port', required=False, default=6677)
    parser.add_argument('--ip-address', type=str, help='ip address', required=False, default='225.1.1.1')
    """
        keep this for wired headset + cyton
        -2 for streaming board
        2 cyton
        1 ganglion
        0 for cyton 8 chann
    """
    parser.add_argument('--board-id', type=int, help='board id, check docs to get a list of supported boards',
                        required=False, default=BID)
    """
        -1 synthetic: 8channels, 250Hz
    """
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD.value)
    args = parser.parse_args()
    args.sampling_rate=250
    putParams(args)

def putParams(args):
    global board
    params = BrainFlowInputParams()
    params.ip_port = args.ip_port
    params.serial_port = args.serial_port
    params.mac_address = args.mac_address
    params.other_info = args.other_info
    params.serial_number = args.serial_number
    params.ip_address = args.ip_address
    params.ip_protocol = args.ip_protocol
    params.timeout = args.timeout
    params.file = args.file
    params.master_board = args.master_board

    board = BoardShim(args.board_id, params)

def check_dir():
    """
    Add by Anson
    Date: 18/7/2026
    Changes: create daily csv for recording, create dataset/ if not exist
    """
    #project_dir/EEG/data_receive.py -> project_dir
    proj_dir = Path(__file__).resolve().parents[1]
    dataset_dir = proj_dir / "dataset"
    #create if not exist
    dataset_dir.mkdir(parents=True, exist_ok=True)
    csv_path = dataset_dir / f"{dt.date.today()}.csv" 
    return csv_path.as_posix()

def start(BID=-2, port=4, plot_domain="t"):
    """
    Edit by Anson
    Date: 18/7/2026
    Changes: added parameters for board id and port for data_record, allow csv output
    """
    global board, outlet, domain
    config(BID, port)
    domain = plot_domain
    board.prepare_session()
    info = StreamInfo("StringMarkers", "Markers", 1, 0, "string", "uid")
    outlet = StreamOutlet(info)
    #put into project_dir/dataset/XXXX.csv
    board.start_stream(45000, f"file://{check_dir()}:a")

def end():
    global board
    if board is not None and board.is_prepared():
        board.stop_stream()
        board.release_session()
    board = None

def put_marker(word, phase="start"):
    """
    Edit by Anson
    Date: 18/7/2026
    Changes: to mark phase of marker, if end then +1000
    """
    global board, outlet
    if len(word) == 1:
        value = float(ord(word))
        if phase == "end":
            value += 1000
        board.insert_marker(value)
    else:
        outlet.push_sample([f"{phase}:{word}"])

def get_BID():
    """
    Add by Anson
    Date: 18/7/2026
    Changes: for virtual board and real board distinction
    """
    if args.board_id == BoardIds.STREAMING_BOARD.value:
        return args.master_board
    return args.board_id

def plot_data(block=True):
    #plot frequency domain
    global board, args
    #index of eeg channels
    eeg_chann = BoardShim.get_eeg_channels(get_BID())[:8]
    print(eeg_chann)

    fig, axes = plt.subplots(4, 2, figsize=(10, 8))
    axes_flat = axes.flatten() # Make it easy to iterate
        
    # Pre-create line objects for better performance
    lines = []
    for i in range(len(eeg_chann)):
        line, = axes_flat[i].plot([], [], lw=1)
        if(domain=="f"):
            axes_flat[i].set_title(f'CH {i+1} (FFT)', fontsize=9)
            axes_flat[i].set_xlabel('Frequency (Hz)', fontsize=8)
            axes_flat[i].set_ylabel('Magnitude', fontsize=8)
            axes_flat[i].grid(True, alpha=0.3)
            axes_flat[i].set_xlim(0, 50) # Frequency range 0-50 Hz
            axes_flat[i].set_ylim(0, 200) # Initial Y-axis limit for FFT magnitude
        else:
            axes_flat[i].set_title(f'CH {i+1}', fontsize=8)
            axes_flat[i].grid(True, alpha=0.3)
            # Set a default Y-axis limit for EEG (typically +/- 100-500 uV)
            axes_flat[i].set_ylim(-200, 200) 
        lines.append(line)

    plt.tight_layout()

    def update(frame):
        # Get last 1000 samples (4 seconds at 250Hz)
        data = cl.clean(board.get_current_board_data(1000), eeg_chann, args.sampling_rate)
            
        if data.size > 0:
            N = len(data)
            for i, channel_idx in enumerate(eeg_chann):
                if(domain=="f"):
                    y = data[channel_idx].values
                    # Compute Real Fast Fourier Transform (RFFT)
                    ##rfft discard negative as EEG signals are real numbers, get 501 bins for N = 1000, 
                    ##abs value of complex numbers, raw unnormalized sum of sine amplitudes
                    fft_vals = np.abs(np.fft.rfft(y))
                    ##generate x axis freq array per time interval(1/250)
                    freqs = np.fft.rfftfreq(len(y), 1 / args.sampling_rate)
                    
                    # Convert raw FFT magnitudes to Peak Amplitude in uV
                    ##get average amplitude by /N, *2 as rfft discarded negative
                    fft_uV_peak = fft_vals * (2.0 / N)
                    ##index 0 = 0Hz(baseline voltage of chann)
                    fft_uV_peak[0] = fft_vals[0] / N
                    #if N is even, last bin sit at 125Hz
                    if len(y) % 2 == 0:
                        fft_uV_peak[-1] = fft_vals[-1] / len(y)
                    #get root mean square of voltage
                    fft_uV_rms = fft_uV_peak / np.sqrt(2.0)
                    # Update line data for FFT plot in uV
                    lines[i].set_data(freqs, fft_uV_rms)
                    
                    # Auto-scale Y-axis in uV (typical EEG peaks are 5 - 30 uV)
                    max_val = np.max(fft_uV_rms) if len(fft_uV_rms) > 0 else 20
                    axes_flat[i].set_ylim(0, max(20, max_val * 1.1))
                    axes_flat[i].set_ylabel('Amplitude (uV)', fontsize=8)
                else:
                    # Update the Y-data for each line
                    lines[i].set_data(range(data.shape[0]), data[channel_idx])
                        
                    # Optional: Auto-scale X-axis to match the data length
                    axes_flat[i].set_xlim(0, data.shape[0])

            return lines

    ani = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)
    #set blocking mechanism
    plt.show(block=block)
    # Return animation reference so it stays alive in non-blocking mode
    return ani

if __name__ == "__main__":
    #print(os.path.dirname(__file__))
    start()
    try:
        plot_data()
    except KeyboardInterrupt:
        pass
    end()