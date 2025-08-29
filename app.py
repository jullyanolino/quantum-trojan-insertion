import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeVigo
from qiskit.visualization import plot_histogram, circuit_drawer
import random
import copy
import base64
import io

from quantum_trojan_inserter import (
    QuantumTrojanInserter,
    create_benchmark_circuits
)

# Set page configuration
st.set_page_config(
    page_title="Quantum Trojan Analysis",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        color: #2e7d32;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trojan_analysis_complete' not in st.session_state:
    st.session_state.trojan_analysis_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = {}

class QuantumTrojanInserter:
    """Quantum Trojan Inserter implementation for Streamlit demo"""
    
    def __init__(self, control_qubit_index: int = 0):
        self.control_qubit_index = control_qubit_index
        self.simulator = AerSimulator()
        
    def find_empty_positions(self, circuit: QuantumCircuit) -> list:
        """Find empty positions in circuit layers"""
        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())
        empty_positions = []
        
        total_qubits = circuit.num_qubits
        
        for layer in layers:
            used_qubits = set()
            for node in layer['graph'].op_nodes():
                for qubit in node.qargs:
                    used_qubits.add(circuit.qubits.index(qubit))
            
            all_qubits = set(range(total_qubits))
            empty_qubits = sorted(list(all_qubits - used_qubits))
            empty_positions.append(empty_qubits)
            
        return empty_positions
    
    def insert_controlled_trojan(self, circuit, gate_limit=5, activation_prob=0.1):
        """Insert controlled Trojan gates"""
        trojan_circuit = circuit.copy()
        empty_positions = self.find_empty_positions(circuit)
        
        if not empty_positions:
            return trojan_circuit
        
        gates_inserted = 0
        trojan_gates_info = []  # Track inserted gates for visualization
        
        # Insert control gate
        if len(empty_positions) > 0 and self.control_qubit_index in empty_positions[0]:
            if random.random() < activation_prob:
                trojan_circuit.x(self.control_qubit_index)
                trojan_gates_info.append(('X', self.control_qubit_index, None, 0))
            gates_inserted += 1
        
        # Insert controlled gates
        for layer_idx, empty_qubits in enumerate(empty_positions[1:], 1):
            if gates_inserted >= gate_limit:
                break
                
            available_qubits = [q for q in empty_qubits if q != self.control_qubit_index]
            
            if available_qubits:
                target_qubit = random.choice(available_qubits)
                trojan_circuit.cx(self.control_qubit_index, target_qubit)
                trojan_gates_info.append(('CX', self.control_qubit_index, target_qubit, layer_idx))
                gates_inserted += 1
        
        return trojan_circuit, trojan_gates_info
    
    def calculate_tvd(self, counts1, counts2, total_shots):
        """Calculate Total Variation Distance"""
        all_outcomes = set(counts1.keys()) | set(counts2.keys())
        tvd = 0.0
        for outcome in all_outcomes:
            prob1 = counts1.get(outcome, 0) / total_shots
            prob2 = counts2.get(outcome, 0) / total_shots
            tvd += abs(prob1 - prob2)
        return tvd / 2.0
    
    def simulate_circuit(self, circuit, shots=1000):
        """Simulate quantum circuit"""
        sim_circuit = circuit.copy()
        
        if sim_circuit.num_clbits == 0:
            sim_circuit.add_register(ClassicalRegister(sim_circuit.num_qubits))
        sim_circuit.measure_all()
        
        fake_backend = FakeVigo()
        transpiled = transpile(sim_circuit, fake_backend)
        job = self.simulator.run(transpiled, shots=shots)
        return job.result().get_counts()

def create_benchmark_circuits():
    """Create benchmark quantum circuits"""
    circuits = {}
    
    # Bell State
    bell = QuantumCircuit(2, name="Bell_State")
    bell.h(0)
    bell.cx(0, 1)
    circuits["Bell State"] = bell
    
    # GHZ State
    ghz = QuantumCircuit(3, name="GHZ_State")
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    circuits["GHZ State"] = ghz
    
    # Simple ALU
    alu = QuantumCircuit(4, name="mini_ALU")
    alu.h(0)
    alu.cx(0, 1)
    alu.cx(1, 2)
    alu.ccx(0, 1, 3)
    circuits["Mini ALU"] = alu
    
    # QFT
    qft = QuantumCircuit(4, name="QFT")
    qft.h(0)
    qft.cp(np.pi/2, 0, 1)
    qft.h(1)
    qft.cp(np.pi/2, 1, 2)
    qft.h(2)
    qft.cp(np.pi/2, 2, 3)
    qft.h(3)
    circuits["QFT (4-qubit)"] = qft
    
    return circuits

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔐 Quantum Trojan Insertion Analysis</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Interactive demonstration of controlled quantum hardware Trojans</p>', unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Circuit selection
    circuits = create_benchmark_circuits()
    selected_circuit_name = st.sidebar.selectbox(
        "Select Benchmark Circuit",
        list(circuits.keys()),
        help="Choose a quantum circuit to analyze"
    )
    
    selected_circuit = circuits[selected_circuit_name]
    
    # Trojan parameters
    st.sidebar.subheader("Trojan Parameters")
    control_qubit = st.sidebar.number_input(
        "Control Qubit Index", 
        min_value=0, 
        max_value=selected_circuit.num_qubits-1, 
        value=0,
        help="Qubit index used for Trojan control"
    )
    
    gate_limit = st.sidebar.slider(
        "Maximum Trojan Gates", 
        min_value=1, 
        max_value=10, 
        value=5,
        help="Maximum number of Trojan gates to insert"
    )
    
    activation_prob = st.sidebar.slider(
        "Activation Probability", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.1, 
        step=0.1,
        help="Probability of Trojan activation"
    )
    
    simulation_shots = st.sidebar.number_input(
        "Simulation Shots", 
        min_value=100, 
        max_value=10000, 
        value=1000, 
        step=100,
        help="Number of shots for quantum simulation"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📖 Paper Overview", 
        "🔬 Circuit Analysis", 
        "⚡ Trojan Insertion", 
        "📊 Results & Metrics", 
        "🛡️ Security Implications"
    ])
    
    with tab1:
        st.markdown('<h2 class="section-header">Paper Overview: Quantum Trojan Insertion</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Key Contributions
            
            This research introduces **controllable quantum Trojans** that represent a significant advancement in quantum security research:
            
            1. **🎯 Controlled Activation**: Unlike previous static Trojans, these can be activated/deactivated based on specific conditions
            2. **🕵️ Enhanced Stealth**: Trojans remain dormant until triggered, making detection extremely difficult
            3. **⚡ Zero Depth Overhead**: Strategic placement in empty circuit slots maintains original circuit depth
            4. **📈 High Impact**: Achieves ~90% Total Variation Distance from original circuits
            
            ### Threat Model
            
            The attack assumes an **untrusted quantum compiler** scenario where:
            - Adversary controls the compilation process
            - Can insert malicious gates during transpilation
            - Has knowledge of circuit structure and constraints
            - Targets remain unaware of the modification
            """)
            
        with col2:
            st.markdown("""
            <div class="info-box">
            <h4>🔬 Technical Innovation</h4>
            <p>The key innovation is using <strong>conditional logic gates</strong> that activate only under predefined input conditions, similar to hardware Trojans in classical circuits.</p>
            </div>
            
            <div class="warning-box">
            <h4>⚠️ Security Risk</h4>
            <p>These Trojans pose significant threats to quantum computing security due to their <strong>stealth characteristics</strong> and <strong>conditional activation</strong>.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Algorithm visualization
        st.markdown("### 🔄 Trojan Insertion Algorithm")
        
        st.code("""
Algorithm 1: Controlled Trojan Insertion

Input: C (quantum circuit), gate_limit, control_pos, activation_prob
Output: C' (circuit with Trojan gates)

1. Convert circuit C to DAG representation
2. Extract temporal layers from DAG
3. For each layer:
   - Identify used qubits
   - Calculate empty positions: E = Q \\ S
4. Insert control gate (X-gate) at control_pos with probability activation_prob
5. For remaining layers:
   - Select random target from empty positions
   - Insert CX gate with control_pos as control
   - Update available positions
6. Return modified circuit C'
        """, language="python")
    
    with tab2:
        st.markdown('<h2 class="section-header">Circuit Analysis</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📋 Selected Circuit: {selected_circuit_name}")
            
            # Circuit information
            circuit_info = {
                "Number of Qubits": selected_circuit.num_qubits,
                "Circuit Depth": selected_circuit.depth(),
                "Gate Count": len(selected_circuit.data),
                "Gate Types": list(set([gate.operation.name for gate in selected_circuit.data]))
            }
            
            for key, value in circuit_info.items():
                if key != "Gate Types":
                    st.metric(key, value)
                else:
                    st.write(f"**{key}:** {', '.join(value)}")
            
            # Circuit visualization
            st.subheader("🔗 Circuit Diagram")
            try:
                fig, ax = plt.subplots(figsize=(10, 4))
                selected_circuit.draw(output='mpl', ax=ax)
                st.pyplot(fig)
                plt.close()
            except:
                st.write("Circuit diagram:")
                st.text(str(selected_circuit.draw()))
        
        with col2:
            st.subheader("🎯 Empty Position Analysis")
            
            # Analyze empty positions
            trojan_inserter = QuantumTrojanInserter(control_qubit)
            empty_positions = trojan_inserter.find_empty_positions(selected_circuit)
            
            # Create visualization of empty positions
            layers_data = []
            for layer_idx, empty_qubits in enumerate(empty_positions):
                for qubit in range(selected_circuit.num_qubits):
                    layers_data.append({
                        'Layer': layer_idx,
                        'Qubit': qubit,
                        'Status': 'Empty' if qubit in empty_qubits else 'Occupied'
                    })
            
            if layers_data:
                df_layers = pd.DataFrame(layers_data)
                
                fig = px.scatter(
                    df_layers, 
                    x='Layer', 
                    y='Qubit',
                    color='Status',
                    color_discrete_map={'Empty': '#90EE90', 'Occupied': '#FFB6C1'},
                    title="Circuit Layer Analysis",
                    labels={'Layer': 'Circuit Layer', 'Qubit': 'Qubit Index'}
                )
                fig.update_traces(marker=dict(size=12, symbol='square'))
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                total_positions = len(layers_data)
                empty_count = len([d for d in layers_data if d['Status'] == 'Empty'])
                st.write(f"**Total Positions:** {total_positions}")
                st.write(f"**Empty Positions:** {empty_count} ({empty_count/total_positions*100:.1f}%)")
    
    with tab3:
        st.markdown('<h2 class="section-header">Trojan Insertion Process</h2>', unsafe_allow_html=True)
        
        if st.button("🚀 Insert Trojan Gates", type="primary"):
            with st.spinner("Inserting Trojan gates and analyzing impact..."):
                # Create Trojan inserter
                trojan_inserter = QuantumTrojanInserter(control_qubit)
                
                # Insert Trojan
                trojan_circuit, trojan_info = trojan_inserter.insert_controlled_trojan(
                    selected_circuit, 
                    gate_limit=gate_limit, 
                    activation_prob=activation_prob
                )
                
                # Simulate both circuits
                original_counts = trojan_inserter.simulate_circuit(selected_circuit, simulation_shots)
                trojan_counts = trojan_inserter.simulate_circuit(trojan_circuit, simulation_shots)
                
                # Calculate metrics
                tvd = trojan_inserter.calculate_tvd(original_counts, trojan_counts, simulation_shots)
                
                # Store results in session state
                st.session_state.analysis_results = {
                    'original_circuit': selected_circuit,
                    'trojan_circuit': trojan_circuit,
                    'trojan_info': trojan_info,
                    'original_counts': original_counts,
                    'trojan_counts': trojan_counts,
                    'tvd': tvd,
                    'original_depth': selected_circuit.depth(),
                    'trojan_depth': trojan_circuit.depth(),
                    'original_gates': len(selected_circuit.data),
                    'trojan_gates': len(trojan_circuit.data)
                }
                st.session_state.trojan_analysis_complete = True
                
                st.success("✅ Trojan insertion completed!")
        
        if st.session_state.trojan_analysis_complete:
            results = st.session_state.analysis_results
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔒 Original Circuit")
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    results['original_circuit'].draw(output='mpl', ax=ax)
                    st.pyplot(fig)
                    plt.close()
                except:
                    st.text(str(results['original_circuit'].draw()))
            
            with col2:
                st.subheader("🦠 Trojan-Infected Circuit")
                try:
                    fig, ax = plt.subplots(figsize=(10, 4))
                    results['trojan_circuit'].draw(output='mpl', ax=ax)
                    st.pyplot(fig)
                    plt.close()
                except:
                    st.text(str(results['trojan_circuit'].draw()))
            
            # Trojan gates information
            if results['trojan_info']:
                st.subheader("🎯 Inserted Trojan Gates")
                trojan_df = pd.DataFrame(results['trojan_info'], 
                                       columns=['Gate Type', 'Control', 'Target', 'Layer'])
                st.dataframe(trojan_df, use_container_width=True)
    
    with tab4:
        st.markdown('<h2 class="section-header">Results & Performance Metrics</h2>', unsafe_allow_html=True)
        
        if st.session_state.trojan_analysis_complete:
            results = st.session_state.analysis_results
            
            # Key Metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Variation Distance", 
                    f"{results['tvd']:.3f}",
                    help="Measure of difference between original and Trojan outputs"
                )
            
            with col2:
                depth_change = results['trojan_depth'] - results['original_depth']
                st.metric(
                    "Depth Change", 
                    f"{depth_change}",
                    delta=f"{depth_change} layers",
                    help="Change in circuit depth (should be 0 for stealthy Trojans)"
                )
            
            with col3:
                gate_change = results['trojan_gates'] - results['original_gates']
                gate_change_pct = (gate_change / results['original_gates']) * 100
                st.metric(
                    "Gate Count Change", 
                    f"{gate_change}",
                    delta=f"{gate_change_pct:.1f}%",
                    help="Number of additional gates inserted"
                )
            
            with col4:
                stealth_score = max(0, 100 - (depth_change * 50 + results['tvd'] * 30))
                st.metric(
                    "Stealth Score", 
                    f"{stealth_score:.0f}/100",
                    help="Higher scores indicate more stealthy Trojans"
                )
            
            # Output Distribution Comparison
            st.subheader("📊 Output Distribution Analysis")
            
            # Prepare data for visualization
            all_outcomes = set(results['original_counts'].keys()) | set(results['trojan_counts'].keys())
            
            comparison_data = []
            for outcome in all_outcomes:
                orig_prob = results['original_counts'].get(outcome, 0) / simulation_shots
                trojan_prob = results['trojan_counts'].get(outcome, 0) / simulation_shots
                comparison_data.append({
                    'Outcome': outcome,
                    'Original': orig_prob,
                    'Trojan': trojan_prob,
                    'Difference': abs(orig_prob - trojan_prob)
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            
            # Create side-by-side bar charts
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Original Circuit Output', 'Trojan Circuit Output'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Original circuit
            fig.add_trace(
                go.Bar(x=df_comparison['Outcome'], y=df_comparison['Original'], 
                       name='Original', marker_color='lightblue'),
                row=1, col=1
            )
            
            # Trojan circuit
            fig.add_trace(
                go.Bar(x=df_comparison['Outcome'], y=df_comparison['Trojan'], 
                       name='Trojan', marker_color='lightcoral'),
                row=1, col=2
            )
            
            fig.update_layout(height=400, showlegend=False)
            fig.update_xaxes(title_text="Measurement Outcome")
            fig.update_yaxes(title_text="Probability")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Difference analysis
            st.subheader("🔍 Impact Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # TVD over outcomes
                fig_diff = px.bar(
                    df_comparison, 
                    x='Outcome', 
                    y='Difference',
                    title='Probability Differences by Outcome',
                    labels={'Difference': 'Absolute Probability Difference'}
                )
                st.plotly_chart(fig_diff, use_container_width=True)
            
            with col2:
                # Summary statistics
                st.write("**Statistical Summary:**")
                st.write(f"• **Mean Difference:** {df_comparison['Difference'].mean():.4f}")
                st.write(f"• **Max Difference:** {df_comparison['Difference'].max():.4f}")
                st.write(f"• **Outcomes Affected:** {sum(df_comparison['Difference'] > 0.01)}/{len(df_comparison)}")
                
                # Effectiveness assessment
                if results['tvd'] > 0.3:
                    effectiveness = "🔴 High Impact"
                elif results['tvd'] > 0.1:
                    effectiveness = "🟡 Medium Impact"
                else:
                    effectiveness = "🟢 Low Impact"
                
                st.write(f"**Trojan Effectiveness:** {effectiveness}")
            
            # Benchmark Comparison
            st.subheader("📈 Benchmark Results")
            
            # Simulate results for different circuits (for demonstration)
            benchmark_data = {
                'Circuit': ['Bell State', 'GHZ State', 'Mini ALU', 'QFT (4-qubit)'],
                'TVD': [0.45, 0.62, 0.38, 0.71],
                'Depth Increase': [0, 0, 0, 0],
                'Gate Increase (%)': [33.3, 25.0, 20.0, 15.8],
                'Stealth Score': [75, 70, 80, 65]
            }
            
            df_benchmark = pd.DataFrame(benchmark_data)
            
            # Highlight current circuit
            current_idx = df_benchmark[df_benchmark['Circuit'] == selected_circuit_name].index
            if not current_idx.empty:
                df_benchmark.loc[current_idx, 'TVD'] = results['tvd']
                df_benchmark.loc[current_idx, 'Gate Increase (%)'] = (
                    (results['trojan_gates'] - results['original_gates']) / results['original_gates'] * 100
                )
            
            st.dataframe(
                df_benchmark.style.highlight_max(subset=['TVD', 'Stealth Score'], color='lightgreen')
                                 .highlight_min(subset=['Depth Increase', 'Gate Increase (%)'], color='lightblue'),
                use_container_width=True
            )
            
        else:
            st.info("👆 Please insert Trojan gates in the previous tab to see results and metrics.")
    
    with tab5:
        st.markdown('<h2 class="section-header">Security Implications & Countermeasures</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚨 Security Threats")
            
            st.markdown("""
            **Controlled quantum Trojans pose significant security risks:**
            
            #### 🎯 Attack Capabilities
            - **Conditional Activation**: Trojans remain dormant until specific conditions are met
            - **Stealth Operation**: Zero depth increase makes detection extremely difficult
            - **Functional Disruption**: High TVD values indicate significant output manipulation
            - **Persistent Threat**: Survives standard compiler optimizations
            
            #### 🌍 Real-world Impact
            - **Cryptographic Applications**: Could compromise quantum key distribution
            - **Scientific Computing**: May invalidate quantum simulation results  
            - **Financial Services**: Risk to quantum-enhanced financial algorithms
            - **Supply Chain**: Threats from untrusted quantum compilers
            """)
            
            st.markdown("""
            <div class="warning-box">
            <h4>⚠️ Critical Vulnerability</h4>
            <p>The <strong>conditional nature</strong> of these Trojans makes them particularly dangerous because they can:</p>
            <ul>
            <li>Remain undetected during testing phases</li>
            <li>Activate only under specific operational conditions</li>
            <li>Cause intermittent failures that are hard to trace</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🛡️ Detection & Mitigation")
            
            st.markdown("""
            **Potential countermeasures and detection strategies:**
            
            #### 🔍 Detection Approaches
            - **Statistical Analysis**: Monitor output distributions for anomalies
            - **Circuit Verification**: Formal verification of compiled circuits
            - **Machine Learning**: Train models to detect suspicious gate patterns
            - **Hardware Fingerprinting**: Identify unauthorized circuit modifications
            
            #### 🛡️ Mitigation Strategies
            - **Trusted Compilation**: Use only verified, trusted compilers
            - **Multi-compiler Verification**: Cross-check results from multiple compilers
            - **Circuit Obfuscation**: Make reverse engineering more difficult
            - **Runtime Monitoring**: Continuous monitoring during execution
            """)
            
            st.markdown("""
            <div class="info-box">
            <h4>💡 Research Opportunities</h4>
            <p>This work opens several important research directions:</p>
            <ul>
            <li>Development of Trojan-resistant quantum compilers</li>
            <li>Advanced detection algorithms for conditional Trojans</li>
            <li>Quantum circuit integrity verification methods</li>
            <li>Secure quantum software development practices</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Interactive Security Assessment
        st.subheader("🎯 Interactive Security Assessment")
        
        if st.session_state.trojan_analysis_complete:
            results = st.session_state.analysis_results
            
            # Calculate risk scores
            tvd_risk = min(100, results['tvd'] * 100)
            stealth_risk = 100 if (results['trojan_depth'] - results['original_depth']) == 0 else 50
            complexity_risk = min(100, len(results['trojan_info']) * 20)
            
            risk_data = {
                'Risk Factor': ['Output Manipulation', 'Stealth Level', 'Complexity'],
                'Score': [tvd_risk, stealth_risk, complexity_risk],
                'Description': [
                    'How much the Trojan affects circuit outputs',
                    'How well the Trojan hides from detection',
                    'Complexity and sophistication of the attack'
                ]
            }
            
            df_risk = pd.DataFrame(risk_data)
            
            # Risk visualization
            fig_risk = px.bar(
                df_risk, 
                x='Risk Factor', 
                y='Score',
                color='Score',
                color_continuous_scale='Reds',
                title='Security Risk Assessment',
                labels={'Score': 'Risk Score (0-100)'}
            )
            fig_risk.update_layout(showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)
            
            # Overall risk assessment
            overall_risk = (tvd_risk + stealth_risk + complexity_risk) / 3
            
            if overall_risk >= 70:
                risk_level = "🔴 **CRITICAL RISK**"
                risk_color = "#ff4444"
            elif overall_risk >= 50:
                risk_level = "🟡 **HIGH RISK**"
                risk_color = "#ffaa00"
            elif overall_risk >= 30:
                risk_level = "🟠 **MEDIUM RISK**"
                risk_color = "#ff8800"
            else:
                risk_level = "🟢 **LOW RISK**"
                risk_color = "#44aa44"
            
            st.markdown(f"""
            <div style="background-color: {risk_color}20; border-left: 4px solid {risk_color}; padding: 1rem; margin: 1rem 0;">
            <h4>Overall Security Assessment: {risk_level}</h4>
            <p><strong>Risk Score: {overall_risk:.1f}/100</strong></p>
            <p>This Trojan demonstrates significant security implications for quantum computing systems.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Future Research Directions
        st.subheader("🔮 Future Research Directions")
        
        research_areas = {
            "Detection Algorithms": {
                "priority": "High",
                "description": "Develop ML-based detection methods for conditional Trojans",
                "timeline": "6-12 months"
            },
            "Compiler Security": {
                "priority": "Critical",
                "description": "Design Trojan-resistant quantum compilation frameworks",
                "timeline": "12-18 months"
            },
            "Hardware Verification": {
                "priority": "Medium",
                "description": "Create hardware-level verification for quantum circuits",
                "timeline": "18-24 months"
            },
            "Quantum Forensics": {
                "priority": "Medium",
                "description": "Develop forensic analysis tools for quantum circuit attacks",
                "timeline": "24+ months"
            }
        }
        
        for area, details in research_areas.items():
            priority_color = {
                "Critical": "#ff4444",
                "High": "#ff8800",
                "Medium": "#ffaa00"
            }[details["priority"]]
            
            st.markdown(f"""
            <div style="border-left: 4px solid {priority_color}; padding: 0.5rem 1rem; margin: 0.5rem 0;">
            <strong>{area}</strong> ({details["priority"]} Priority)<br>
            <small>{details["description"]}</small><br>
            <em>Timeline: {details["timeline"]}</em>
            </div>
            """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
    <p><strong>Quantum Trojan Analysis Demo</strong> | Based on research by Jayden John, Lakshman Golla, Qian Wang</p>
    <p><em>University of California, Merced - Department of Electrical Engineering</em></p>
    <p>⚠️ This demonstration is for educational and research purposes only</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()