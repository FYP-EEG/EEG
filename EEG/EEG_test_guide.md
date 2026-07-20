# Guide to run EEG simulation

Record data with VEP, receive EEG signals from OpenBCI GUI, filter signals, and output to a live plot and a CSV.

1. Download OpenBCI GUI from [here](https://github.com/OpenBCI/OpenBCI_GUI/releases/download/v6.0.0-beta.1/openbcigui_v6.0.0-beta.1_windows64.zip).
2. Open OpenBCI GUI.
3. In the top-left dropdown menu, select **Synthetic (algorithmic)**.
4. Set the Synthetic Configs:
   * 8 channel
   * IP: `225.1.1.1`
   * Port: `6677`
5. Click **Start Data Stream** (the green button in the top left).
6. Open VS Code in your project directory.
7. Make a virtual environment:
   ```cmd
   py -m venv .venv

```

8. Activate the environment and install dependencies:
```cmd
.venv\Scripts\activate
pip install -r requirements_loose.txt

```


9. Run the following for a simple live plot and signal filtering:
```cmd
py eeg/data_receive.py

```


10. Run the following for flashing words/characters:
```cmd
py eeg/data_record.py

```


11. Type characters while focused on the Pygame window. You can observe the current word displaying in your terminal.
12. Hit `Enter` and the word will appear for 2 seconds. It will have a 1-second buffer time, and only then will it allow input again.
