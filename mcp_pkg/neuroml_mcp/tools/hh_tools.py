#!/usr/bin/env python3
"""
Tools specific to Open Worm HH simulation

File: mcp_pkg/neuroml_mcp/tools/hh_tools.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from dataclasses import asdict
from textwrap import dedent
from typing import Any, Dict

from neuroml_mcp.tools.sandbox.sandbox import RunPythonCode
from .sandbox import nml_mcp_sandbox

sbox = nml_mcp_sandbox


async def run_hh_simulation_tool(
    current_injection: float = 0.1,
    duration: float = 300.0,
    delay: float = 50.0,
    soma_length: float = 20.0,
    soma_diam: float = 20.0,
    temperature: float = 6.3,
) -> Dict[str, Any]:
    """Run a Hodgkin-Huxley single compartment neuron simulation using NEURON.

    Use this tool to simulate a neuron and observe its electrical behaviour
    such as action potentials and firing rate in response to current injection.
    Results include summary statistics and a downsampled voltage trace over time.

    This uses the standard Hodgkin-Huxley squid giant axon model as a proof of
    concept. It is useful for demonstrating how membrane excitability changes
    with different stimulation parameters.

    Inputs:

    - current_injection (float, default 0.1): injected current amplitude in nA.
      Increase to make the neuron fire more frequently.
      Decrease below threshold (~0.05 nA) to observe subthreshold behaviour.
    - duration (float, default 300.0): total simulation duration in milliseconds.
    - delay (float, default 50.0): delay in milliseconds before current injection begins.
      Must be less than duration.
    - soma_length (float, default 20.0): length of the soma compartment in microns.
    - soma_diam (float, default 20.0): diameter of the soma compartment in microns.
    - temperature (float, default 6.3): simulation temperature in Celsius.
      The original Hodgkin-Huxley model was recorded at 6.3C.
      Increasing temperature speeds up channel kinetics.

    Output:

    Dictionary with keys:
    - stdout (str): JSON string containing:
        - firing_rate_hz (float): number of action potentials per second
        - num_action_potentials (int): total action potentials during simulation
        - peak_voltage_mv (float): maximum membrane voltage reached
        - resting_voltage_mv (float): initial resting membrane potential
        - simulation_duration_ms (float): total simulation duration
        - current_injection_na (float): injected current used
        - temperature_c (float): temperature used
        - voltage_trace (dict): downsampled trace with keys t_ms and v_mv
    - stderr (str): any errors from NEURON
    - returncode (int): 0 if successful, non-zero if error occurred
    - data (dict): additional metadata

    Examples:

    - Default simulation: run_hh_simulation_tool()
    - Strong stimulation: run_hh_simulation_tool(current_injection=0.5)
    - Subthreshold (no firing expected): run_hh_simulation_tool(current_injection=0.01)
    - Compare firing rates: call twice with different current_injection values and compare firing_rate_hz
    - Long simulation: run_hh_simulation_tool(duration=1000.0)
    - Temperature effect: call twice with temperature=6.3 and temperature=25.0
    """

    code = dedent(f"""
import json
from neuron import h
import numpy as np

# setup temperature
h.celsius = {temperature}

# create soma compartment
soma = h.Section(name='soma')
soma.L = {soma_length}
soma.diam = {soma_diam}

# insert standard Hodgkin-Huxley mechanism (built into NEURON)
soma.insert('hh')

# current clamp stimulus
stim = h.IClamp(soma(0.5))
stim.delay = {delay}
stim.dur = {duration - delay}
stim.amp = {current_injection}

# recording vectors
v_vec = h.Vector().record(soma(0.5)._ref_v)
t_vec = h.Vector().record(h._ref_t)

# run simulation
h.finitialize(-65)
h.continuerun({duration})

# convert to plain lists
t = list(t_vec)
v = list(v_vec)

# count action potentials by upward zero-crossings of membrane voltage
# crossing 0mV going upward reliably identifies action potential peaks in HH
crossings = sum(
    1 for i in range(1, len(v))
    if v[i-1] < 0 and v[i] >= 0
)

stimulus_duration_s = ({duration} - {delay}) / 1000.0
firing_rate = crossings / stimulus_duration_s if stimulus_duration_s > 0 else 0

# output as JSON to stdout for LLM to parse
# voltage trace downsampled by factor of 10 to reduce token usage
print(json.dumps({{
    "firing_rate_hz": round(firing_rate, 2),
    "num_action_potentials": crossings,
    "peak_voltage_mv": round(max(v), 2),
    "resting_voltage_mv": round(v[0], 2),
    "simulation_duration_ms": {duration},
    "current_injection_na": {current_injection},
    "temperature_c": {temperature},
    "voltage_trace": {{
        "t_ms": t[::10],
        "v_mv": v[::10]
    }}
}}))
""")

    request = RunPythonCode(code=code)
    async with sbox(".") as f:
        result = await f.run(request)
    return asdict(result)






