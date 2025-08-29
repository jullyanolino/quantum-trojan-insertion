# Quantum Trojan Insertion: Interactive Analysis Demo

This repository contains an implementation and interactive demonstration of the quantum Trojan insertion technique described in the paper **"Quantum Trojan Insertion: Controlled Activation for Covert Circuit Manipulation"** by Jayden John, Lakshman Golla, and Qian Wang from UC Merced.

## 🔬 Paper Overview

The research introduces a novel class of **controllable quantum Trojans** that can be activated or deactivated under specific conditions. Unlike previous static quantum Trojans, these sophisticated attacks:

- ✅ Remain dormant until triggered by predefined input conditions
- ✅ Maintain zero circuit depth overhead for enhanced stealth
- ✅ Achieve ~90% Total Variation Distance from original circuits
- ✅ Resist detection through standard optimization techniques

## 🚀 Features

### Interactive Streamlit Demo
- **📊 Real-time Analysis**: Visualize Trojan insertion and impact analysis
- **🔧 Configurable Parameters**: Adjust gate limits, activation probability, and control qubits
- **📈 Performance Metrics**: Calculate Total Variation Distance and circuit overhead
- **🛡️ Security Assessment**: Interactive security risk evaluation
- **📚 Educational Content**: Comprehensive explanations of the technique

### Implementation Highlights
- **🎯 Controlled Activation**: Implements the paper's conditional logic approach
- **⚡ Strategic Placement**: Identifies empty circuit slots for stealth insertion
- **📊 Comprehensive Analysis**: TVD calculation and performance evaluation
- **🔍 Visualization**: Circuit diagrams and impact comparisons

## 📁 File Structure

```
quantum-trojan-demo/
├── quantum_trojan_inserter.py    # Core implementation (main module)
├── app.py                        # Streamlit interactive demo
├── example_usage.py              # Comprehensive analysis examples
├── requirements.txt              # Python dependencies
├── README.md                     # This documentation
└── trojan_analysis_results.csv   # Generated results (after running examples)
```

## 🔧 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**:
```bash
git clone <repository-url>
cd quantum-trojan-demo
```

2. **Create a virtual environment** (recommended):
```bash
python -m venv quantum_trojan_env
source quantum_trojan_env/bin/activate  # On Windows: quantum_trojan_env\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## 📱 Usage

### Running the Streamlit Demo

1. **Start the application**:
```bash
streamlit run app.py
```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Explore the demo**:
   - Select benchmark quantum circuits
   - Configure Trojan parameters
   - Insert and analyze Trojan gates
   - Review security implications

### Using the Core Implementation

```python
from quantum_trojan_inserter import QuantumTrojanInserter, create_benchmark_circuits

# Create a Trojan inserter
trojan_inserter = QuantumTrojanInserter(control_qubit_index=0)

# Load benchmark circuits
circuits = create_benchmark_circuits()
bell_circuit = circuits["Bell State"]

# Insert controlled Trojan
trojan_circuit, trojan_info = trojan_inserter.insert_controlled_trojan(
    bell_circuit, 
    gate_limit=5, 
    activation_prob=0.1
)

# Analyze impact
original_counts = trojan_inserter.simulate_circuit(bell_circuit)
trojan_counts = trojan_inserter.simulate_circuit(trojan_circuit)

tvd = trojan_inserter.calculate_tvd(original_counts, trojan_counts, 1000)
print(f"Total Variation Distance: {tvd:.3f}")
```

## 📊 Demo Sections

### 1. 📖 Paper Overview
- Key contributions and innovations
- Threat model explanation
- Algorithm visualization

### 2. 🔬 Circuit Analysis
- Circuit parameter analysis
- Empty position identification
- Visual circuit representation

### 3. ⚡ Trojan Insertion
- Interactive Trojan insertion
- Side-by-side circuit comparison
- Inserted gate tracking

### 4. 📊 Results & Metrics
- Total Variation Distance calculation
- Output distribution comparison
- Performance overhead analysis
- Benchmark comparisons

### 5. 🛡️ Security Implications
- Risk assessment
- Detection strategies
- Mitigation approaches
- Future research directions

## 🔬 Technical Implementation

### Core Components

#### `QuantumTrojanInserter` Class
- **`find_empty_positions()`**: Analyzes circuit layers to identify insertion points
- **`insert_controlled_trojan()`**: Implements the strategic Trojan insertion algorithm
- **`calculate_tvd()`**: Computes Total Variation Distance between distributions
- **`simulate_circuit()`**: Runs quantum simulations with realistic noise models

#### Algorithm Details
1. **Circuit Analysis**: Convert quantum circuit to DAG representation
2. **Layer Extraction**: Identify temporal layers and gate dependencies
3. **Empty Position Detection**: Find unused qubit positions in each layer
4. **Strategic Insertion**: Place control and target gates in optimal locations
5. **Impact Assessment**: Measure functional and performance impact

### Benchmark Circuits
- **Bell State**: 2-qubit entanglement circuit
- **GHZ State**: 3-qubit multipartite entanglement
- **Mini ALU**: 4-qubit arithmetic logic unit
- **QFT**: 4-qubit Quantum Fourier Transform

## 📈 Key Metrics

### Total Variation Distance (TVD)
Measures the statistical difference between original and Trojan circuit outputs:

```
TVD = (1/2) * Σ |P_original(x) - P_trojan(x)|
```

### Stealth Metrics
- **Depth Overhead**: Should be 0 for perfect stealth
- **Gate Count Increase**: Typically ~20% increase
- **Detection Resistance**: Based on optimization survival

## ⚠️ Security Implications

### Critical Threats
- **Conditional Activation**: Trojans activate only under specific conditions
- **Supply Chain Risk**: Untrusted compilers can inject malicious code
- **Detection Evasion**: Advanced stealth techniques resist standard detection
- **Persistent Attacks**: Survive compiler optimization passes

### Potential Targets
- Quantum cryptography systems
- Financial quantum algorithms
- Scientific quantum simulations
- Government quantum applications

## 🛡️ Defensive Measures

### Detection Strategies
- **Statistical Monitoring**: Track output distribution anomalies
- **Multi-compiler Verification**: Cross-check compilation results
- **ML-based Detection**: Train models on circuit patterns
- **Formal Verification**: Mathematical proof of circuit correctness

### Mitigation Approaches
- **Trusted Compilation**: Use verified compilers only
- **Circuit Obfuscation**: Make reverse engineering difficult
- **Runtime Monitoring**: Continuous execution monitoring
- **Hardware Verification**: Physical-level integrity checks

## 🔮 Future Research

### Priority Areas
1. **Detection Algorithms**: ML-based conditional Trojan detection
2. **Compiler Security**: Trojan-resistant compilation frameworks
3. **Hardware Verification**: Circuit integrity at the hardware level
4. **Quantum Forensics**: Post-attack analysis and attribution

## 📚 References

**Primary Paper**: 
- John, J., Golla, L., & Wang, Q. "Quantum Trojan Insertion: Controlled Activation for Covert Circuit Manipulation." *arXiv preprint arXiv:2502.08880* (2025).

**Related Work**:
- Das, S., & Ghosh, S. "Trojan attacks on variational quantum circuits and countermeasures." *ISQED 2024*.
- Roy, R., Das, S., & Ghosh, S. "Hardware trojans in quantum circuits, their impacts, and defense." *ISQED 2024*.

## 📄 License

This educational demonstration is provided for research and educational purposes. Please respect quantum security research ethics and use responsibly.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests to improve the demonstration.

## ⚠️ Disclaimer

This demonstration is for **educational and research purposes only**. It should not be used for malicious purposes. The authors and contributors are not responsible for any misuse of this technology.

---

**Contact**: For questions about the implementation or research, please refer to the original paper's authors at UC Merced.