"""Safe helpers and explicit acquisition profiles for OpenBCI Cyton.

The OpenBCI GUI and this project both acquire Cyton samples through BrainFlow.
The important difference is therefore the ADS1299 channel profile, not a hidden
GUI-only packet decoder.  Two profiles are kept deliberately separate:

``openbci_gui_default``
    GUI Reset/Default parity for ordinary passive electrodes: gain x24, Normal
    input, BIAS included, SRB2 connected, SRB1 disconnected.

``thinkpulse_active``
    Powered ThinkPulse sensors: gain x8 by default (x8 or lower), Normal input,
    BIAS excluded, SRB2 connected, SRB1 disconnected.  A physical BIAS lead is
    not expected; one earlobe reference is expected on SRB2.

Do not silently exchange these profiles.  They encode different electrode and
reference assumptions.  A gain override changes only the amplifier gain; it
does not change BIAS/SRB routing.

BrainFlow's Python ``config_board`` binding sends commands through its native
library and then decodes an optional response as UTF-8.  Some Cyton/driver
combinations return a non-text byte after a successful command.  The helper
below tolerates that response-decoding failure only; subsequent raw-signal tests
remain the source of truth.
"""

CYTON_GAIN_CODES = {1: "0", 2: "1", 4: "2", 6: "3", 8: "4", 12: "5", 24: "6"}
ELECTRODE_MODES = ("passive", "thinkpulse")
CYTON_PROFILE_NAMES = ("openbci_gui_default", "thinkpulse_active")

CYTON_PROFILES = {
    "openbci_gui_default": {
        "description": "OpenBCI GUI Reset/Default parity for passive electrodes",
        "electrode_mode": "passive",
        "gain": 24,
        "maximum_gain": 24,
        "input": "normal",
        "include_bias": True,
        "srb2": True,
        "srb1": False,
        "physical_bias_expected": True,
        "reference": "SRB2",
    },
    "thinkpulse_active": {
        "description": "Powered ThinkPulse active-electrode profile",
        "electrode_mode": "thinkpulse",
        "gain": 8,
        "maximum_gain": 8,
        "input": "normal",
        "include_bias": False,
        "srb2": True,
        "srb1": False,
        "physical_bias_expected": False,
        "reference": "SRB2",
    },
}


def normalize_electrode_mode(value):
    mode = str(value or "passive").strip().lower().replace("-", "")
    aliases = {"active": "thinkpulse", "openbcithinkpulse": "thinkpulse"}
    mode = aliases.get(mode, mode)
    if mode not in ELECTRODE_MODES:
        raise ValueError(f"unknown electrode mode {value!r}; expected {ELECTRODE_MODES}")
    return mode


def default_cyton_profile(electrode_mode="passive"):
    """Return the explicit profile name matching a legacy electrode mode."""
    mode = normalize_electrode_mode(electrode_mode)
    return ("thinkpulse_active" if mode == "thinkpulse"
            else "openbci_gui_default")


def normalize_cyton_profile(value=None, *, electrode_mode=None):
    """Normalize profile aliases while preserving old ``electrode_mode`` APIs."""
    if value is None or not str(value).strip():
        return default_cyton_profile(electrode_mode or "passive")
    profile = str(value).strip().lower().replace("-", "_")
    aliases = {
        "passive": "openbci_gui_default",
        "gui": "openbci_gui_default",
        "gui_default": "openbci_gui_default",
        "openbci_gui": "openbci_gui_default",
        "thinkpulse": "thinkpulse_active",
        "active": "thinkpulse_active",
    }
    profile = aliases.get(profile, profile)
    if profile not in CYTON_PROFILES:
        raise ValueError(
            f"unknown Cyton profile {value!r}; expected {CYTON_PROFILE_NAMES}")
    return profile


def cyton_profile_settings(profile=None, *, electrode_mode=None, gain=None):
    """Return one validated profile with an optional gain-only override."""
    name = normalize_cyton_profile(profile, electrode_mode=electrode_mode)
    settings = dict(CYTON_PROFILES[name])
    settings["name"] = name
    resolved_gain = settings["gain"] if gain is None else int(gain)
    if resolved_gain not in CYTON_GAIN_CODES:
        raise ValueError(
            f"unsupported Cyton gain {resolved_gain}; expected {tuple(CYTON_GAIN_CODES)}")
    if resolved_gain > int(settings["maximum_gain"]):
        if name == "thinkpulse_active":
            raise ValueError("ThinkPulse requires Cyton gain 8 or lower")
        raise ValueError(
            f"profile {name} allows gain {settings['maximum_gain']} or lower")
    settings["gain"] = resolved_gain
    return settings


def channel_settings_command(channel, gain=24, include_bias=True,
                             srb2=True, srb1=False):
    """Build one documented Cyton ``x...X`` Normal-input channel command."""
    channel = int(channel)
    gain = int(gain)
    if channel not in range(1, 9):
        raise ValueError("Cyton channel must be 1..8")
    if gain not in CYTON_GAIN_CODES:
        raise ValueError(f"unsupported Cyton gain {gain}; expected {tuple(CYTON_GAIN_CODES)}")
    # x CH POWER_DOWN GAIN INPUT BIAS SRB2 SRB1 X
    return (f"x{channel}0{CYTON_GAIN_CODES[gain]}0"
            f"{int(bool(include_bias))}{int(bool(srb2))}{int(bool(srb1))}X")


def live_channel_commands(electrode_mode="passive", gain=None, *, profile=None):
    """Return eight Normal-input commands for an explicit acquisition profile.

    ``electrode_mode`` is retained for compatibility.  New callers should pass
    ``profile=`` so reports record exactly which GUI-parity or active-electrode
    assumptions were applied.
    """
    settings = cyton_profile_settings(
        profile, electrode_mode=electrode_mode, gain=gain)
    return tuple(channel_settings_command(
        channel, gain=settings["gain"],
        include_bias=settings["include_bias"],
        srb2=settings["srb2"], srb1=settings["srb1"])
        for channel in range(1, 9))


def send_cyton_command(board, command):
    """Send known ASCII and tolerate optional response-text decode failure only."""
    value = str(command)
    try:
        value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise ValueError("Cyton commands must contain ASCII only") from exc
    if not value:
        raise ValueError("Cyton command cannot be empty")
    try:
        response = board.config_board(value)
        return {"response": response, "response_decode_ignored": False}
    except UnicodeDecodeError as exc:
        return {
            "response": None,
            "response_decode_ignored": True,
            "decode_error": str(exc),
        }
