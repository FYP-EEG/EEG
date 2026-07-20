Guide to run EEG simulation: record data with VEP, EEG signal receive from OpenBCI GUI, signal filtering, and output to a live plot and a csv
<ol>
    <li>Download OpenBCI GUI from <a href="https://github.com/OpenBCI/OpenBCI_GUI/releases/download/v6.0.0-beta.1/openbcigui_v6.0.0-beta.1_windows64.zip">here</a></li>
    <li>open OpenBCI GUI</li>
    <li>In top left dropdown menu, select synthetic(algorithmic)</li>
    <li>Synthetic Configs
        <ul>
            <li>8 channel</li>
            <li>IP: 225.1.1.1</li>
            <li>Port: 6677</li>
        </ul>
    </li>
    <li>Click start data stream in top left green button</li>
    <li>Open vscode</li>
    <li>Make virtual environment with <code>py -m venv .venv</code></li>
    <li>Run <code>
        .venv\Scripts\activate
        <br>
        pip install -r requirements_loose.txt
    </code></li>
    <li>Run <code>py eeg/data_receive.py</code> for simple live plot and signal filtering</li>
    <li>Run <code>py eeg/data_record.py</code> for flashing words/characters</li>
    <li>type characters while focused on the pygame window, you can observe current word to display in cmd</li>
    <li>Then hit enter and word will appear for 2 seconds, and will have 1s buffer time and only then it allow input again</li>
</ol>