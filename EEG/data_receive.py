import argparse
import time

from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
#py data_receive.py --board-id -2 --ip-address 225.1.1.1 --ip-port 6677 --master-board -1

def config():
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
    parser.add_argument('--serial-port', type=str, help='serial port', required=False, default='')
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
                        required=False, default=BoardIds.STREAMING_BOARD.value)
    """
        -1 synthetic: 8channels, 250Hz
    """
    parser.add_argument('--master-board', type=int, help='master board id for streaming and playback boards',
                        required=False, default=BoardIds.SYNTHETIC_BOARD.value)
    args = parser.parse_args()
    board = putParams(args)

    return board, args

def putParams(args):
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
    return board

def main():
    board, args = config()
    board.prepare_session()
    board.start_stream()
    #index of eeg channels
    eeg_chann = BoardShim.get_eeg_channels(args.master_board)[:8]
    print(eeg_chann)

    fig, axes = plt.subplots(4,2)
    axes_flat = axes.flatten() # Make it easy to iterate
        
    # Pre-create line objects for better performance
    lines = []
    for i in range(len(eeg_chann)):
        line, = axes_flat[i].plot([], [], lw=1)
        axes_flat[i].set_title(f'CH {i+1}', fontsize=8)
        axes_flat[i].grid(True, alpha=0.3)
        # Set a default Y-axis limit for EEG (typically +/- 100-500 uV)
        axes_flat[i].set_ylim(-200, 200) 
        lines.append(line)

    plt.tight_layout()

    def update(frame):
        # Get last 1000 samples (4 seconds at 250Hz)
        data = board.get_current_board_data(1000)
            
        if data.size > 0:
            for i, channel_idx in enumerate(eeg_chann):
                # Update the Y-data for each line
                lines[i].set_data(range(data.shape[1]), data[channel_idx])
                    
                # Optional: Auto-scale X-axis to match the data length
                axes_flat[i].set_xlim(0, data.shape[1])

            return lines

    ani = FuncAnimation(fig, update, interval=60, blit=False, cache_frame_data=False)#update every 50ms
    plt.show()
    board.stop_stream()
    board.release_session()

if __name__ == "__main__":
    main()