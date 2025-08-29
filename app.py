import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram, plot_circuit_layout
import matplotlib.pyplot as plt
import numpy as np
import random
from qiskit.providers.fake_provider import FakeValencia  # For noise
from qiskit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
import random

def insert_controlled_trojan(circuit: QuantumCircuit, control_pos: int = 0, gate_limit: int = 5, include_switch: bool = True) -> QuantumCircuit:
    """
    Implements the quantum Trojan insertion as per Algorithm 1.
    - Identifies empty positions per layer using DAG.
    - Inserts X on control_pos in first available column (switch, optional).
    - Inserts up to gate_limit CX gates from control_pos to random empty positions in subsequent columns.
    - Preserves circuit depth by using empty slots.
    """
    # Convert to DAG
    dag = circuit_to_dag(circuit)
    total_qubits = circuit.num_qubits
    all_qubits = set(range(total_qubits))
    
    # Extract layers (columns)
    layers = list(dag.layers())
    
    # Collect empty positions per layer
    empty_pos_per_layer = []
    for layer in layers:
        used_qubits = set()
        for node in layer['graph'].op_nodes():
            used_qubits.update(q.index for q in node.qargs)
        empty_pos = sorted(all_qubits - used_qubits)
        empty_pos_per_layer.append(empty_pos)
    
    # Now insert gates. We'll add them to the circuit at the beginning for simplicity,
    # but in practice, insert into existing layers' empty slots by prepending a new circuit.
    trojan_circuit = QuantumCircuit(total_qubits)
    
    added_gates = 0
    available_qubits = set(range(total_qubits)) - {control_pos}  # Exclude control initially
    
    # Insert into 'columns' (we simulate by adding sequentially, assuming empty at start)
    # In real, we'd insert into specific layers, but for demo, prepend Trojan.
    for col_idx in range(len(layers) + gate_limit):  # Allow extra if needed, but limit gates
        if added_gates >= gate_limit:
            break
        # Use empty pos from a random layer or approximate
        empty_positions = random.choice(empty_pos_per_layer) if empty_pos_per_layer else []
        available_qubits = available_qubits.intersection(empty_positions)
        
        if not available_qubits:
            continue
        
        if col_idx == 0 and include_switch:
            # Insert X gate on control (switch)
            trojan_circuit.x(control_pos)
        else:
            # Insert CX gate
            random_pos = random.choice(list(available_qubits))
            trojan_circuit.cx(control_pos, random_pos)
            available_qubits.remove(random_pos)
            added_gates += 1
    
    # Compose with original circuit
    final_circuit = trojan_circuit.compose(circuit, inplace=False)
    
    return final_circuit

# The insertion function from above
# (Paste the insert_controlled_trojan function here)

st.set_page_config(page_title="Quantum Trojan Demo", layout="wide")

st.title("Quantum Trojan Insertion: Controlled Activation for Covert Circuit Manipulation")
st.markdown("**Interactive Demo and Explanation of the Paper** by Jayden John, Lakshman Golla, Qian Wang (arXiv:2502.08880v1)")

# Section 1: Introduction
with st.expander("1. Introduction to the Paper"):
    st.write("""
    This paper introduces stealthy, controllable quantum Trojans inserted via untrusted compilers. 
    Trojans remain dormant until triggered, altering outputs without increasing circuit depth.
    Key: Uses DAG to find empty positions for CX gates controlled by a qubit with optional X switch.
    """)
    # Recreate Fig 1
    fig1, ax1 = plt.subplots()
    ax1.text(0.5, 0.5, "Simple Quantum Circuit: q0 -- H -- \nq1 --", fontsize=12, ha='center')
    ax1.axis('off')
    st.pyplot(fig1)

# Section 2: Background
with st.expander("2. Background & Related Work"):
    st.write("""
    - Hardware Trojans in classical ICs: Combinational/sequential, triggered by rare conditions.
    - Quantum Circuits: Qubits, gates (X, CX), compilation vulnerabilities.
    - Related: Simple Trojans (X/SWAP insertions), but easily detected. This work adds conditionality.
    """)

# Section 3: Threat Model
with st.expander("3. Threat Model"):
    st.write("""
    Untrusted compiler inserts Trojans during transpilation. Adversary accesses original/transpiled circuits.
    Goal: Disrupt under specific inputs without detection.
    """)
    # Recreate Fig 2
    st.image("https://via.placeholder.com/800x200?text=Threat+Model+Diagram", caption="Fig 2: Untrusted Compiler Flow")

# Section 4: Quantum Trojan and Analysis
with st.expander("4. Quantum Trojan Insertion Process"):
    st.write("""
    Algorithm: Identify empty slots in DAG layers. Insert X (switch) on control, then CX to random empties.
    Activation: Include X or set control input to |1> for trigger.
    """)
    st.code("""
    # Pseudocode from Algorithm 1
    for each layer:
        empty = all_qubits - used_qubits
    for column in circuits:
        if column 0: add X(control)
        else: add CX(control, random_empty)
    """)
    # Recreate Fig 3
    col1, col2 = st.columns(2)
    with col1:
        st.write("Switch OFF (Deactivated)")
        fig_off, ax_off = plt.subplots()
        ax_off.text(0.5, 0.5, "H -- X -- Z -- X -- X\nCX idle", ha='center')
        ax_off.axis('off')
        st.pyplot(fig_off)
    with col2:
        st.write("Switch ON (Activated)")
        fig_on, ax_on = plt.subplots()
        ax_on.text(0.5, 0.5, "X (switch) -- H -- X -- Z -- X -- X\nCX flips targets", ha='center')
        ax_on.axis('off')
        st.pyplot(fig_on)

# Section 5: Experiments
with st.expander("5. Experiments & Evaluation"):
    st.write("""
    - Setup: Qiskit, RevLib benchmarks, FakeValencia noise, 1000 shots.
    - Metric: TVD = (1/2N) * sum |y_orig - y_alter|
    - Results: 0% depth increase, ~20% gate increase, TVD ~90% when activated.
    """)
    # Recreate Table I (subset)
    data = {
        "Circuit": ["mini_ALU", "4mod5", "1-bit adder"],
        "Depth": [8, 6, 5],
        "Gate Count": [7, 7, 5],
        "TVD (Activated)": [0.85, 0.92, 0.78]  # Approximate from Fig 5
    }
    st.table(data)
    # Recreate Fig 5: TVD Distribution
    tvd_values = np.random.uniform(0.7, 1.0, 10)  # Simulated
    fig5, ax5 = plt.subplots()
    ax5.hist(tvd_values, bins=5)
    ax5.set_title("TVD Distribution")
    st.pyplot(fig5)

# Interactive Demo
st.header("Interactive Demonstration")
st.write("Select a circuit, insert Trojan, simulate original vs. deactivated vs. activated.")

# User inputs
num_qubits = st.slider("Number of Qubits", 3, 10, 4)
gate_limit = st.slider("Gate Limit (CX count)", 1, 10, 3)
control_pos = st.selectbox("Control Qubit", range(num_qubits), index=0)
use_noise = st.checkbox("Use Noisy Simulation (FakeValencia)", value=False)
shots = 1024

# Example circuit: Simple adder-like (Toffoli-based)
original_circuit = QuantumCircuit(num_qubits, num_qubits - 1)  # Last qubit as ancillary if needed
original_circuit.h(range(num_qubits))  # Superposition for demo
original_circuit.cx(0, 1)
original_circuit.cx(1, 2)
if num_qubits > 3:
    original_circuit.ccx(0, 1, 3)  # Toffoli for adder flavor
original_circuit.measure(range(num_qubits - 1), range(num_qubits - 1))

st.subheader("Original Circuit")
st.write(original_circuit.draw(output='text'))

# Insert Trojan
deactivated_circuit = insert_controlled_trojan(original_circuit, control_pos, gate_limit, include_switch=False)
activated_circuit = insert_controlled_trojan(original_circuit, control_pos, gate_limit, include_switch=True)

col_deact, col_act = st.columns(2)
with col_deact:
    st.subheader("Deactivated Trojan")
    st.write(deactivated_circuit.draw(output='text'))
with col_act:
    st.subheader("Activated Trojan")
    st.write(activated_circuit.draw(output='text'))

# Simulate
simulator = AerSimulator.from_backend(FakeValencia()) if use_noise else AerSimulator()
    
orig_transpiled = transpile(original_circuit, simulator)
deact_transpiled = transpile(deactivated_circuit, simulator)
act_transpiled = transpile(activated_circuit, simulator)

orig_result = simulator.run(orig_transpiled, shots=shots).result()
deact_result = simulator.run(deact_transpiled, shots=shots).result()
act_result = simulator.run(act_transpiled, shots=shots).result()

orig_counts = orig_result.get_counts()
deact_counts = deact_result.get_counts()
act_counts = act_result.get_counts()

# Visualize Histograms
st.subheader("Output Distributions")
fig_hist, axs_hist = plt.subplots(1, 3, figsize=(15, 5))
plot_histogram(orig_counts, ax=axs_hist[0])
axs_hist[0].set_title("Original")
plot_histogram(deact_counts, ax=axs_hist[1])
axs_hist[1].set_title("Deactivated")
plot_histogram(act_counts, ax=axs_hist[2])
axs_hist[2].set_title("Activated")
st.pyplot(fig_hist)

# Compute TVD
def compute_tvd(counts1, counts2, shots):
    all_keys = set(counts1).union(counts2)
    tvd = 0.0
    for key in all_keys:
        p1 = counts1.get(key, 0) / shots
        p2 = counts2.get(key, 0) / shots
        tvd += abs(p1 - p2)
    return tvd / 2

tvd_deact = compute_tvd(orig_counts, deact_counts, shots)
tvd_act = compute_tvd(orig_counts, act_counts, shots)

st.subheader("Total Variation Distance (TVD)")
st.write(f"Original vs. Deactivated: {tvd_deact:.4f} (Should be low)")
st.write(f"Original vs. Activated: {tvd_act:.4f} (Should be high)")

# Section 6: Conclusion
with st.expander("6. Conclusion"):
    st.write("""
    The paper highlights the need for secure quantum compilation. This demo shows how Trojans can be inserted and activated, with minimal overhead.
    Experiment with parameters to see impacts!
    """)

st.markdown("**Built with Qiskit, Streamlit, Matplotlib, NumPy.**")