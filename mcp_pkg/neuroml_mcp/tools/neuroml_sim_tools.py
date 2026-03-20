#!/usr/bin/env python3
"""
NeuroML simulation tools for C. elegans cell models and networks.

Provides tools to:
- Simulate individual NeuroML cell models (GenericNeuronCell, GenericMuscleCell)
- Simulate multi-neuron c302 networks with real connectome data

Uses pyneuroml / jNeuroML to run simulations from NeuroML/LEMS definitions.

File: mcp_pkg/neuroml_mcp/tools/neuroml_sim_tools.py
"""

import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict
from textwrap import dedent
from typing import Any, Dict, List, Optional

from neuroml_mcp.tools.sandbox.sandbox import RunPythonCode
from .sandbox import nml_mcp_sandbox

sbox = nml_mcp_sandbox
logger = logging.getLogger(__name__)

# ── Model registry ───────────────────────────────────────────────────────────
# Maps friendly model names to their NeuroMLlite JSON config files.
# Paths are relative to the models directory.
#
# On HuggingFace Spaces (OpenWormLLM), models/ lives at /app/models/
# Locally, it defaults to ~/Desktop/OpenWormLLM/models/ but can be
# overridden via the NML_MODELS_DIR environment variable.

_DEFAULT_MODELS_DIR = os.path.expanduser("~/Desktop/OpenWormLLM/models")
MODELS_DIR = os.environ.get("NML_MODELS_DIR", _DEFAULT_MODELS_DIR)

AVAILABLE_MODELS = {
    "GenericNeuronCell": {
        "description": (
            "Generic C. elegans neuron with Leak, k_slow, and ca_simple "
            "channels (parameters C from c302). Single-compartment soma."
        ),
        "cell_file": "GenericNeuronCell.cell.nml",
        "network_file": "IClamp_GenericNeuronCell.json",
        "sim_file": "Sim_IClamp_GenericNeuronCell.json",
        "default_stim_amp": "4pA",
        "channels": {
            "Leak": {"default_cond": "0.05 mS_per_cm2", "ion": "non_specific"},
            "k_slow": {"default_cond": "0.1 mS_per_cm2", "ion": "k"},
            "ca_simple": {"default_cond": "0.06 mS_per_cm2", "ion": "ca"},
        },
    },
    "GenericMuscleCell": {
        "description": (
            "Generic C. elegans muscle cell with Leak, k_slow, and ca_simple "
            "channels (parameters C from c302). Elongated single-compartment."
        ),
        "cell_file": "GenericMuscleCell.cell.nml",
        "network_file": "IClamp_GenericMuscleCell.json",
        "sim_file": "Sim_IClamp_GenericMuscleCell.json",
        "default_stim_amp": "4pA",
        "channels": {
            "Leak": {"default_cond": "0.00105 mS_per_cm2", "ion": "non_specific"},
            "k_slow": {"default_cond": "0.3 mS_per_cm2", "ion": "k"},
            "ca_simple": {"default_cond": "0.25 mS_per_cm2", "ion": "ca"},
        },
    },
}


def _check_pynml_available() -> bool:
    """Check if pyneuroml is importable."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import pyneuroml"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


def _check_c302_available() -> bool:
    """Check if c302 is importable."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import c302"],
            capture_output=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


# ── c302 circuit presets ──────────────────────────────────────────────────────
# Common neuron groups for quick access. Users can also specify arbitrary
# neuron lists.

C302_CIRCUIT_PRESETS = {
    "forward_locomotion": {
        "description": "Forward locomotion command circuit",
        "neurons": ["AVBL", "AVBR", "DB1", "DB2", "DB3", "VB1", "VB2", "VB3"],
        "stimulate": ["AVBL", "AVBR"],
    },
    "backward_locomotion": {
        "description": "Backward locomotion command circuit",
        "neurons": ["AVAL", "AVAR", "DA1", "DA2", "DA3", "VA1", "VA2", "VA3"],
        "stimulate": ["AVAL", "AVAR"],
    },
    "touch_response": {
        "description": "Gentle touch response circuit (anterior)",
        "neurons": ["ALML", "ALMR", "AVM", "AVDL", "AVDR", "AVAL", "AVAR",
                     "AVBL", "AVBR", "DA1", "DB1", "VA1", "VB1"],
        "stimulate": ["ALML", "ALMR"],
    },
    "pharyngeal": {
        "description": "Pharyngeal nervous system",
        "neurons": ["M1", "M2L", "M2R", "M3L", "M3R", "M4", "M5",
                     "I1L", "I1R", "I2L", "I2R", "I3", "I4", "I5", "I6",
                     "MI", "NSML", "NSMR", "MCL", "MCR"],
        "stimulate": ["M1", "M3R", "M4", "M5", "I1L", "I4"],
    },
    "thermotaxis": {
        "description": "Thermosensory circuit (AFD → AIY → AIZ)",
        "neurons": ["AFDL", "AFDR", "AIYL", "AIYR", "AIZL", "AIZR",
                     "RIAL", "RIAR", "RIML", "RIMR"],
        "stimulate": ["AFDL", "AFDR"],
    },
    "chemotaxis": {
        "description": "Chemosensory decision circuit (AWC, AWA → AIY, AIB, AIZ)",
        "neurons": ["AWCL", "AWCR", "AWAL", "AWAR", "AIYL", "AIYR",
                     "AIBL", "AIBR", "AIZL", "AIZR"],
        "stimulate": ["AWCL", "AWCR"],
    },
}


async def list_neuroml_models_tool() -> Dict[str, Any]:
    """List available NeuroML cell models for simulation.

    Returns a dictionary of model names with their descriptions,
    available ion channels, and default parameters.

    Output:
    Dictionary with keys:
    - models (dict): model_name → {description, channels, default_stim_amp}
    - models_dir (str): path to models directory
    - models_dir_exists (bool): whether the directory exists on disk
    """
    models_info = {}
    for name, meta in AVAILABLE_MODELS.items():
        models_info[name] = {
            "description": meta["description"],
            "channels": meta["channels"],
            "default_stim_amp": meta["default_stim_amp"],
        }
    return {
        "models": models_info,
        "models_dir": MODELS_DIR,
        "models_dir_exists": os.path.isdir(MODELS_DIR),
    }


async def run_neuroml_cell_simulation_tool(
    cell_model: str = "GenericNeuronCell",
    stim_amplitude_pA: float = 4.0,
    stim_delay_ms: float = 500.0,
    stim_duration_ms: float = 2000.0,
    sim_duration_ms: float = 3000.0,
    conductance_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Run a NeuroML cell simulation using c302-style C. elegans models.

    Simulates an individual neuron or muscle cell from the OpenWorm c302
    framework with configurable current injection and optional conductance
    scaling.  Uses pyneuroml + jNeuroML to run the LEMS simulation.

    Inputs:

    - cell_model (str, default "GenericNeuronCell"):
      Which cell model to simulate. Options: "GenericNeuronCell", "GenericMuscleCell"

    - stim_amplitude_pA (float, default 4.0):
      Stimulus current amplitude in picoamps.  The c302 cells use pA-scale
      currents (much smaller than the nA used by the HH squid model).

    - stim_delay_ms (float, default 500.0):
      Delay before stimulus onset in milliseconds.

    - stim_duration_ms (float, default 2000.0):
      Duration of the current injection in milliseconds.

    - sim_duration_ms (float, default 3000.0):
      Total simulation duration in milliseconds.  Must be > stim_delay + stim_duration.

    - conductance_overrides (dict, optional):
      Scale factors for ion channel conductance densities.
      Example: {"k_slow": 2.0, "ca_simple": 0.5}
      This multiplies the default conductance by the given factor.
      Available channels depend on the model (see list_neuroml_models_tool).

    Output:

    Dictionary with keys:
    - stdout (str): JSON string containing simulation results:
        - peak_voltage_mv (float)
        - min_voltage_mv (float)
        - resting_voltage_mv (float)
        - mean_voltage_during_stim_mv (float)
        - cell_model (str)
        - stim_amplitude_pA (float)
        - sim_duration_ms (float)
        - conductance_overrides (dict)
        - plot_base64 (str): base64-encoded voltage trace PNG
        - plot_path (str): path to saved PNG file
        - voltage_trace (dict): with t_ms and v_mv arrays
    - stderr (str)
    - returncode (int)
    - data (dict)

    Examples:

    - Default neuron: run_neuroml_cell_simulation_tool()
    - Muscle cell: run_neuroml_cell_simulation_tool(cell_model="GenericMuscleCell")
    - Stronger stimulus: run_neuroml_cell_simulation_tool(stim_amplitude_pA=10.0)
    - Double calcium conductance:
        run_neuroml_cell_simulation_tool(conductance_overrides={"ca_simple": 2.0})
    """
    # ── Validate inputs ──────────────────────────────────────────────────
    if cell_model not in AVAILABLE_MODELS:
        return {
            "stdout": "",
            "stderr": (
                f"Unknown cell model '{cell_model}'. "
                f"Available: {list(AVAILABLE_MODELS.keys())}"
            ),
            "returncode": 1,
            "data": {"error": "unknown_model"},
        }

    if not os.path.isdir(MODELS_DIR):
        return {
            "stdout": "",
            "stderr": (
                f"Models directory not found: {MODELS_DIR}. "
                f"Set NML_MODELS_DIR environment variable to the correct path."
            ),
            "returncode": 1,
            "data": {"error": "models_dir_missing"},
        }

    if not _check_pynml_available():
        return {
            "stdout": "",
            "stderr": (
                "pyneuroml is not installed. Install with: pip install pyneuroml"
            ),
            "returncode": 1,
            "data": {"error": "pyneuroml_not_installed"},
        }

    model_meta = AVAILABLE_MODELS[cell_model]
    cond_overrides = conductance_overrides or {}

    # ── Build simulation Python code ─────────────────────────────────────
    # This runs inside the sandbox subprocess.  It:
    # 1. Uses NeuroMLlite to load the network/sim JSON configs
    # 2. Modifies stimulus parameters
    # 3. Optionally scales channel conductances by modifying the .nml XML
    # 4. Runs via pynml (jNeuroML)
    # 5. Parses output .dat files
    # 6. Generates a matplotlib plot
    # 7. Outputs JSON to stdout

    code = dedent(f"""
import json, os, sys, shutil, tempfile, base64
import xml.etree.ElementTree as ET

# work in a temp directory so we don't pollute the models dir
tmpdir = tempfile.mkdtemp(prefix="nml_sim_")
models_src = "{MODELS_DIR}"

# copy all model files to temp dir
for f in os.listdir(models_src):
    src = os.path.join(models_src, f)
    if os.path.isfile(src):
        shutil.copy2(src, tmpdir)

os.chdir(tmpdir)

# ── 1. Modify stimulus parameters in network JSON ───────────────────
network_file = "{model_meta['network_file']}"
with open(network_file, "r") as nf:
    network_config = json.load(nf)

net_key = list(network_config.keys())[0]
network_config[net_key]["parameters"]["stim_amp"] = "{stim_amplitude_pA}pA"
network_config[net_key]["temperature"] = 34.0

# update stimulus timing
for inp_key, inp_val in network_config[net_key].get("input_sources", {{}}).items():
    inp_val["parameters"]["delay"] = "{stim_delay_ms}ms"
    inp_val["parameters"]["duration"] = "{stim_duration_ms}ms"

with open(network_file, "w") as nf:
    json.dump(network_config, nf, indent=4)

# ── 2. Modify simulation duration ────────────────────────────────────
sim_file = "{model_meta['sim_file']}"
with open(sim_file, "r") as sf:
    sim_config = json.load(sf)

sim_key = list(sim_config.keys())[0]
sim_config[sim_key]["duration"] = {sim_duration_ms}

with open(sim_file, "w") as sf:
    json.dump(sim_config, sf, indent=4)

# ── 3. Apply conductance overrides to .nml cell file ─────────────────
cond_overrides = {json.dumps(cond_overrides)}
cell_file = "{model_meta['cell_file']}"

if cond_overrides:
    import re
    with open(cell_file, 'r') as cf:
        xml_content = cf.read()

    for chan_name, scale_factor in cond_overrides.items():
        # Match: ionChannel="chan_name" condDensity="0.05 mS_per_cm2"
        # or:    condDensity="0.05 mS_per_cm2" ionChannel="chan_name"
        # Use regex to find channelDensity elements for this channel
        pattern = r'(ionChannel\\s*=\\s*"' + re.escape(chan_name) + r'"[^>]*condDensity\\s*=\\s*")([0-9.eE+-]+)(\\s+\\w+)(")'
        def _scale(m):
            new_val = float(m.group(2)) * scale_factor
            return m.group(1) + str(new_val) + m.group(3) + m.group(4)
        xml_content = re.sub(pattern, _scale, xml_content)

        # Also handle reverse attribute order
        pattern2 = r'(condDensity\\s*=\\s*")([0-9.eE+-]+)(\\s+\\w+)("[^>]*ionChannel\\s*=\\s*"' + re.escape(chan_name) + r'")'
        def _scale2(m):
            new_val = float(m.group(2)) * scale_factor
            return m.group(1) + str(new_val) + m.group(3) + m.group(4)
        xml_content = re.sub(pattern2, _scale2, xml_content)

    with open(cell_file, 'w') as cf:
        cf.write(xml_content)

# ── 4. Run simulation via neuromllite + pynml ────────────────────────
# Redirect stdout during neuromllite calls to suppress verbose output
# which would otherwise pollute our JSON stdout.
import io as _io
_old_stdout = sys.stdout
sys.stdout = _io.StringIO()
try:
    from neuromllite.utils import load_simulation_json
    from neuromllite.NetworkGenerator import generate_and_run
    sim = load_simulation_json(sim_file)
    generate_and_run(sim, simulator="jNeuroML")
finally:
    sys.stdout = _old_stdout

# ── 5. Extract voltage trace from output .dat files ──────────────────
# jNeuroML writes results as space-separated .dat files in the current dir.
# The filename pattern is typically: Sim_IClamp_GenericNeuronCell.pop_*.dat
import glob
dat_files = sorted(glob.glob("*.dat"))

t_ms = []
v_mv = []

if dat_files:
    # Use the first .dat file (single-cell simulation has one trace)
    dat_file = dat_files[0]
    with open(dat_file, 'r') as df:
        for line in df:
            parts = line.strip().split()
            if len(parts) >= 2:
                t_s = float(parts[0])   # time in seconds
                v_v = float(parts[1])   # voltage in volts (SI)
                t_ms.append(t_s * 1000.0)   # → milliseconds
                v_mv.append(v_v * 1000.0)   # → millivolts
else:
    print(json.dumps({{"error": "No .dat output files found after simulation"}}))
    sys.exit(0)

if not t_ms:
    print(json.dumps({{"error": "Voltage trace is empty"}}))
    sys.exit(0)

# ── 6. Compute summary statistics ────────────────────────────────────
stim_start_ms = {stim_delay_ms}
stim_end_ms = {stim_delay_ms} + {stim_duration_ms}

# indices during stimulus period
stim_indices = [i for i, ti in enumerate(t_ms) if stim_start_ms <= ti <= stim_end_ms]
v_during_stim = [v_mv[i] for i in stim_indices] if stim_indices else v_mv

peak_v = max(v_mv)
min_v = min(v_mv)
resting_v = v_mv[0] if v_mv else -50.0
mean_v_stim = sum(v_during_stim) / len(v_during_stim) if v_during_stim else 0.0

# ── 7. Generate voltage trace plot ───────────────────────────────────
plot_base64 = ""
plot_path = ""
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ms, v_mv, color='#2563eb', linewidth=0.8)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Membrane Potential (mV)')

    # title with summary stats
    model_name = cell_file.replace(".cell.nml", "")
    depol = peak_v - resting_v
    ax.set_title(
        f'{{model_name}}  |  '
        f'I = {stim_amplitude_pA} pA  |  '
        f'Peak: {{round(peak_v, 1)}} mV  |  '
        f'\\u0394V: {{round(depol, 1)}} mV'
    )

    ax.axhline(y=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvspan(stim_start_ms, stim_end_ms, alpha=0.06, color='orange',
               label=f'Stimulus ({stim_amplitude_pA} pA)')

    # stats annotation box
    cond_text = ""
    if cond_overrides:
        cond_parts = [f"{{k}}: {{v}}x" for k, v in cond_overrides.items()]
        cond_text = "\\nConductance: " + ", ".join(cond_parts)
    stats_text = (
        f"Resting: {{round(resting_v, 1)}} mV\\n"
        f"Peak: {{round(peak_v, 1)}} mV\\n"
        f"Mean (stim): {{round(mean_v_stim, 1)}} mV"
        f"{{cond_text}}"
    )
    ax.text(
        0.02, 0.97, stats_text, transform=ax.transAxes,
        fontsize=7, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='wheat', alpha=0.7)
    )

    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, {sim_duration_ms})
    fig.tight_layout()

    import tempfile as tf
    plot_file = tf.NamedTemporaryFile(
        suffix='.png', prefix='nml_trace_', delete=False,
        dir=os.environ.get('TMPDIR', '/tmp')
    )
    plot_path = plot_file.name
    plot_file.close()
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    with open(plot_path, 'rb') as pf:
        plot_base64 = base64.b64encode(pf.read()).decode('utf-8')
except Exception as plot_err:
    print(f"Warning: plot generation failed: {{plot_err}}", file=sys.stderr)

# ── 8. Output JSON ───────────────────────────────────────────────────
# downsample trace for token efficiency
step = max(1, len(t_ms) // 1000)
print(json.dumps({{
    "peak_voltage_mv": round(peak_v, 3),
    "min_voltage_mv": round(min_v, 3),
    "resting_voltage_mv": round(resting_v, 3),
    "mean_voltage_during_stim_mv": round(mean_v_stim, 3),
    "cell_model": "{cell_model}",
    "stim_amplitude_pA": {stim_amplitude_pA},
    "stim_delay_ms": {stim_delay_ms},
    "stim_duration_ms": {stim_duration_ms},
    "sim_duration_ms": {sim_duration_ms},
    "conductance_overrides": {json.dumps(cond_overrides)},
    "plot_base64": plot_base64,
    "plot_path": plot_path,
    "voltage_trace": {{
        "t_ms": t_ms[::step],
        "v_mv": v_mv[::step]
    }}
}}))

# cleanup temp dir (but leave plot file)
import shutil as sh
try:
    sh.rmtree(tmpdir, ignore_errors=True)
except Exception:
    pass
""")

    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)


# ── c302 network simulation ──────────────────────────────────────────────────


async def list_c302_circuits_tool() -> Dict[str, Any]:
    """List available c302 circuit presets and check c302 availability.

    Returns preset circuit descriptions and whether c302 is installed.

    Output:
    Dictionary with keys:
    - presets (dict): preset_name → {description, neurons, stimulate}
    - c302_available (bool): whether c302 package is installed
    - parameter_sets (list): available c302 parameter sets (A, B, C, D)
    """
    return {
        "presets": C302_CIRCUIT_PRESETS,
        "c302_available": _check_c302_available(),
        "parameter_sets": ["A", "B", "C", "D"],
    }


async def run_c302_network_simulation_tool(
    neurons: List[str],
    neurons_to_stimulate: Optional[List[str]] = None,
    stim_amplitude_pA: float = 5.0,
    stim_delay_ms: float = 100.0,
    stim_duration_ms: float = 300.0,
    sim_duration_ms: float = 500.0,
    parameter_set: str = "C",
    circuit_preset: Optional[str] = None,
    include_muscles: bool = False,
) -> Dict[str, Any]:
    """Simulate a C. elegans neural circuit using the c302 framework.

    Uses real connectome data from the C. elegans wiring diagram to build
    a multi-neuron network with chemical synapses and gap junctions, then
    runs the simulation with jNeuroML.

    Inputs:

    - neurons (list of str):
      List of C. elegans neuron names to include in the simulation.
      Example: ["AVAL", "AVAR", "DA1", "DA2"]
      All 302 neurons are available (e.g. ADAL, AFDL, AIYL, AVAL, AVBL,
      DA1-DA9, DB1-DB7, VA1-VA12, VB1-VB11, etc.)
      Connections between included neurons are extracted from the connectome.

    - neurons_to_stimulate (list of str, optional):
      Which neurons receive current injection.  Defaults to the first neuron.

    - stim_amplitude_pA (float, default 5.0):
      Current injection amplitude in picoamps.

    - stim_delay_ms (float, default 100.0):
      Delay before stimulus onset in milliseconds.

    - stim_duration_ms (float, default 300.0):
      Duration of stimulation in milliseconds.

    - sim_duration_ms (float, default 500.0):
      Total simulation duration in milliseconds.

    - parameter_set (str, default "C"):
      c302 parameter set to use. Options:
      "A" — integrate-and-fire (simplest, fastest)
      "B" — IAF with activity dynamics
      "C" — conductance-based with Leak, k_slow, ca_simple (recommended)
      "D" — conductance-based with additional k_fast channel

    - circuit_preset (str, optional):
      Use a predefined circuit instead of specifying neurons manually.
      Options: "forward_locomotion", "backward_locomotion", "touch_response",
      "pharyngeal", "thermotaxis", "chemotaxis"
      If provided, overrides the neurons and neurons_to_stimulate parameters.

    - include_muscles (bool, default False):
      Whether to include body wall muscles connected to the neurons.

    Output:

    Dictionary with keys:
    - stdout (str): JSON string containing:
        - neurons (dict): per-neuron stats {peak_mv, min_mv, resting_mv, mean_stim_mv}
        - network_info (dict): connections, gap_junctions counts
        - parameter_set (str)
        - stim_amplitude_pA (float)
        - sim_duration_ms (float)
        - plot_base64 (str): base64-encoded multi-trace voltage plot
        - plot_path (str): path to saved PNG
    - stderr (str)
    - returncode (int)
    - data (dict)

    Examples:

    - Backward locomotion preset:
        run_c302_network_simulation_tool([], circuit_preset="backward_locomotion")
    - Custom circuit:
        run_c302_network_simulation_tool(["AVAL", "AVAR", "DA1", "DB1"],
            neurons_to_stimulate=["AVAL"], stim_amplitude_pA=10.0)
    - Pharyngeal circuit with parameter set B:
        run_c302_network_simulation_tool([], circuit_preset="pharyngeal",
            parameter_set="B", sim_duration_ms=1000)
    """
    # ── Resolve circuit preset ──────────────────────────────────────────
    if circuit_preset and circuit_preset in C302_CIRCUIT_PRESETS:
        preset = C302_CIRCUIT_PRESETS[circuit_preset]
        neurons = preset["neurons"]
        if neurons_to_stimulate is None:
            neurons_to_stimulate = preset["stimulate"]

    if not neurons:
        return {
            "stdout": "",
            "stderr": (
                "No neurons specified. Provide a list of neuron names or "
                f"use a circuit_preset: {list(C302_CIRCUIT_PRESETS.keys())}"
            ),
            "returncode": 1,
            "data": {"error": "no_neurons"},
        }

    if neurons_to_stimulate is None:
        neurons_to_stimulate = [neurons[0]]

    # ── Validate dependencies ───────────────────────────────────────────
    if not _check_c302_available():
        return {
            "stdout": "",
            "stderr": (
                "c302 is not installed. Install with: "
                "pip install git+https://github.com/openworm/c302.git"
            ),
            "returncode": 1,
            "data": {"error": "c302_not_installed"},
        }

    if not _check_pynml_available():
        return {
            "stdout": "",
            "stderr": "pyneuroml is not installed. Install with: pip install pyneuroml",
            "returncode": 1,
            "data": {"error": "pyneuroml_not_installed"},
        }

    valid_param_sets = {"A", "B", "C", "D"}
    if parameter_set not in valid_param_sets:
        return {
            "stdout": "",
            "stderr": f"Invalid parameter_set '{parameter_set}'. Use one of: {valid_param_sets}",
            "returncode": 1,
            "data": {"error": "invalid_parameter_set"},
        }

    # ── Build subprocess code ───────────────────────────────────────────
    neurons_json = json.dumps(neurons)
    stim_neurons_json = json.dumps(neurons_to_stimulate)

    code = dedent(f"""
import json, os, sys, tempfile, glob, base64, io, re, subprocess

tmpdir = tempfile.mkdtemp(prefix="c302_sim_")
os.chdir(tmpdir)

# Redirect stdout during c302 verbose output
_old_stdout = sys.stdout
sys.stdout = io.StringIO()
try:
    import c302
    from c302.parameters_{parameter_set} import ParameterisedModel

    params = ParameterisedModel()
    neurons = {neurons_json}
    stim_neurons = {stim_neurons_json}

    nml_doc = c302.generate(
        net_id="c302_{parameter_set}_sim",
        params=params,
        cells=neurons,
        cells_to_stimulate=stim_neurons,
        muscles_to_include={json.dumps([] if not include_muscles else "all")},
        duration={sim_duration_ms},
        dt=0.05,
        target_directory="./",
        verbose=False,
    )

    # Add current injection to stimulated neurons
    for cell in stim_neurons:
        c302.add_new_input(
            nml_doc, cell,
            "{stim_delay_ms}ms", "{stim_duration_ms}ms", "{stim_amplitude_pA}pA",
            params
        )

    # Rewrite .nml with added inputs
    from neuroml.writers import NeuroMLWriter
    NeuroMLWriter.write(nml_doc, "c302_{parameter_set}_sim.net.nml", close=True)
finally:
    sys.stdout = _old_stdout

# ── Count connections from the generated network ─────────────────────
chem_conns = len(nml_doc.networks[0].projections) if nml_doc.networks else 0
elec_conns = len(nml_doc.networks[0].electrical_projections) if nml_doc.networks else 0

# ── Run jNeuroML simulation ─────────────────────────────────────────
env = os.environ.copy()
java_bin = "/opt/homebrew/opt/openjdk/bin"
if os.path.isdir(java_bin) and java_bin not in env.get("PATH", ""):
    env["PATH"] = java_bin + ":" + env.get("PATH", "")

result = subprocess.run(
    [sys.executable, "-m", "pyneuroml.pynml",
     "LEMS_c302_{parameter_set}_sim.xml", "-nogui"],
    capture_output=True, timeout=300, env=env,
)

if result.returncode != 0:
    print(json.dumps({{
        "error": "jNeuroML simulation failed",
        "stderr": result.stderr.decode()[-1000:]
    }}))
    sys.exit(0)

# ── Parse .dat output ────────────────────────────────────────────────
# c302 outputs: c302_X_sim.dat (voltage) and c302_X_sim.activity.dat
# Voltage file columns: time, neuron1_v, neuron2_v, ..., neuron1_ca, neuron2_ca, ...
dat_file = "c302_{parameter_set}_sim.dat"
if not os.path.exists(dat_file):
    dat_files = sorted(glob.glob("*.dat"))
    dat_file = dat_files[0] if dat_files else None

if not dat_file:
    print(json.dumps({{"error": "No output .dat file found"}}))
    sys.exit(0)

# Read LEMS to get column ordering
lems_file = "LEMS_c302_{parameter_set}_sim.xml"
col_names = []
with open(lems_file, "r") as lf:
    lems_content = lf.read()
    # Extract OutputColumn quantities that end in /v (voltage)
    voltage_cols = re.findall(
        r'<OutputColumn\\s+id="([^"]+)"\\s+quantity="([^"]*)/v"',
        lems_content
    )
    col_names = [vc[0].replace("_v", "") for vc in voltage_cols]

if not col_names:
    col_names = neurons

# Parse voltage data
n_neurons = len(col_names)
t_ms_all = []
traces = {{name: [] for name in col_names}}

with open(dat_file, "r") as df:
    for line in df:
        parts = line.strip().split()
        if len(parts) < 1 + n_neurons:
            continue
        t_s = float(parts[0])
        t_ms_all.append(t_s * 1000.0)
        for i, name in enumerate(col_names):
            v_v = float(parts[1 + i])
            traces[name].append(v_v * 1000.0)  # V → mV

if not t_ms_all:
    print(json.dumps({{"error": "Voltage data is empty"}}))
    sys.exit(0)

# ── Per-neuron statistics ────────────────────────────────────────────
stim_start = {stim_delay_ms}
stim_end = {stim_delay_ms} + {stim_duration_ms}
stim_idx = [i for i, t in enumerate(t_ms_all) if stim_start <= t <= stim_end]

neuron_stats = {{}}
for name, v_mv in traces.items():
    v_stim = [v_mv[i] for i in stim_idx] if stim_idx else v_mv
    neuron_stats[name] = {{
        "peak_mv": round(max(v_mv), 3),
        "min_mv": round(min(v_mv), 3),
        "resting_mv": round(v_mv[0], 3),
        "mean_stim_mv": round(sum(v_stim) / len(v_stim), 3) if v_stim else 0.0,
        "stimulated": name in stim_neurons,
    }}

# ── Multi-trace voltage plot ─────────────────────────────────────────
plot_base64 = ""
plot_path = ""
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(col_names)
    colors = plt.cm.tab10.colors if n <= 10 else plt.cm.tab20.colors

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, name in enumerate(col_names):
        color = colors[i % len(colors)]
        lw = 1.2 if name in stim_neurons else 0.8
        ls = "-" if name in stim_neurons else "--"
        ax.plot(t_ms_all, traces[name], color=color, linewidth=lw,
                linestyle=ls, label=name)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Membrane Potential (mV)")
    ax.set_title(
        f"c302 Network (param {parameter_set})  |  "
        f"{{n}} neurons  |  "
        f"I = {stim_amplitude_pA} pA → {{', '.join(stim_neurons)}}"
    )
    ax.axhline(y=0, color="grey", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.axvspan(stim_start, stim_end, alpha=0.06, color="orange",
               label="Stimulus")
    ax.legend(loc="upper right", fontsize=7, ncol=max(1, n // 5))
    ax.set_xlim(0, {sim_duration_ms})
    fig.tight_layout()

    import tempfile as tf
    pf = tf.NamedTemporaryFile(
        suffix=".png", prefix="c302_trace_", delete=False,
        dir=os.environ.get("TMPDIR", "/tmp")
    )
    plot_path = pf.name
    pf.close()
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    with open(plot_path, "rb") as pf:
        plot_base64 = base64.b64encode(pf.read()).decode("utf-8")
except Exception as plot_err:
    print(f"Warning: plot generation failed: {{plot_err}}", file=sys.stderr)

# ── Output JSON ──────────────────────────────────────────────────────
# Downsample traces for token efficiency
step = max(1, len(t_ms_all) // 500)
voltage_traces = {{}}
for name in col_names:
    voltage_traces[name] = traces[name][::step]

print(json.dumps({{
    "neurons": neuron_stats,
    "network_info": {{
        "num_neurons": n_neurons,
        "chemical_projections": chem_conns,
        "electrical_projections": elec_conns,
        "stimulated_neurons": stim_neurons,
    }},
    "parameter_set": "{parameter_set}",
    "stim_amplitude_pA": {stim_amplitude_pA},
    "stim_delay_ms": {stim_delay_ms},
    "stim_duration_ms": {stim_duration_ms},
    "sim_duration_ms": {sim_duration_ms},
    "plot_base64": plot_base64,
    "plot_path": plot_path,
    "voltage_traces": {{
        "t_ms": t_ms_all[::step],
        **voltage_traces,
    }},
}}))

# cleanup
import shutil
try:
    shutil.rmtree(tmpdir, ignore_errors=True)
except Exception:
    pass
""")

    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)
