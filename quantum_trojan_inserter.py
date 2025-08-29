import numpy as np
from qiskit import QuantumCircuit, transpile, ClassicalRegister, QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit_aer import AerSimulator
from qiskit.providers.fake_provider import FakeVigo
import random
from typing import List, Dict, Tuple, Set
import copy

class QuantumTrojanInserter:
    """
    Implementation of controllable quantum Trojan insertion as described in the paper.
    """
    
    def __init__(self, control_qubit_index: int = 0):
        """
        Initialize the Trojan inserter.
        
        Args:
            control_qubit_index: Index of the qubit used as control for Trojan activation
        """
        self.control_qubit_index = control_qubit_index
        self.simulator = AerSimulator()
        
    def find_empty_positions(self, circuit: QuantumCircuit) -> List[List[int]]:
        """
        Find empty positions in each layer of the quantum circuit.
        
        Args:
            circuit: The quantum circuit to analyze
            
        Returns:
            List of lists, where each inner list contains empty qubit indices for that layer
        """
        # Convert circuit to DAG for layer analysis
        dag = circuit_to_dag(circuit)
        layers = list(dag.layers())
        empty_positions = []
        
        total_qubits = circuit.num_qubits
        
        for layer in layers:
            # Get qubits used in this layer
            used_qubits = set()
            for node in layer['graph'].op_nodes():
                for qubit in node.qargs:
                    used_qubits.add(circuit.qubits.index(qubit))
            
            # Find empty positions (complement of used qubits)
            all_qubits = set(range(total_qubits))
            empty_qubits = sorted(list(all_qubits - used_qubits))
            empty_positions.append(empty_qubits)
            
        return empty_positions
    
    def insert_controlled_trojan(self, 
                                circuit: QuantumCircuit, 
                                gate_limit: int = 5,
                                activation_probability: float = 0.1) -> QuantumCircuit:
        """
        Insert controlled Trojan gates into the quantum circuit.
        
        Args:
            circuit: Original quantum circuit
            gate_limit: Maximum number of Trojan gates to insert
            activation_probability: Probability of Trojan activation
            
        Returns:
            Modified circuit with Trojan gates inserted
        """
        # Create a copy of the original circuit
        trojan_circuit = circuit.copy()
        
        # Find empty positions in the circuit
        empty_positions = self.find_empty_positions(circuit)
        
        if not empty_positions:
            return trojan_circuit
        
        gates_inserted = 0
        
        # Insert control gate (X gate) at the beginning if control qubit is available
        if len(empty_positions) > 0 and self.control_qubit_index in empty_positions[0]:
            # Add control gate based on activation probability
            if random.random() < activation_probability:
                trojan_circuit.x(self.control_qubit_index)
            gates_inserted += 1
        
        # Insert controlled gates in subsequent layers
        for layer_idx, empty_qubits in enumerate(empty_positions[1:], 1):
            if gates_inserted >= gate_limit:
                break
                
            # Filter out control qubit from available positions
            available_qubits = [q for q in empty_qubits if q != self.control_qubit_index]
            
            if available_qubits:
                # Randomly select target qubit for CNOT gate
                target_qubit = random.choice(available_qubits)
                
                # Insert controlled-NOT gate with control qubit as control
                trojan_circuit.cx(self.control_qubit_index, target_qubit)
                gates_inserted += 1
                
                # Remove selected qubit from available pool
                available_qubits.remove(target_qubit)
        
        return trojan_circuit
    
    def calculate_total_variation_distance(self, 
                                         counts1: Dict[str, int], 
                                         counts2: Dict[str, int], 
                                         total_shots: int) -> float:
        """
        Calculate Total Variation Distance between two probability distributions.
        
        Args:
            counts1: Measurement counts from first circuit
            counts2: Measurement counts from second circuit
            total_shots: Total number of shots used in simulation
            
        Returns:
            Total Variation Distance value
        """
        # Get all possible outcomes
        all_outcomes = set(counts1.keys()) | set(counts2.keys())
        
        tvd = 0.0
        for outcome in all_outcomes:
            prob1 = counts1.get(outcome, 0) / total_shots
            prob2 = counts2.get(outcome, 0) / total_shots
            tvd += abs(prob1 - prob2)
        
        return tvd / 2.0
    
    def simulate_circuit(self, 
                        circuit: QuantumCircuit, 
                        shots: int = 1000,
                        add_measurements: bool = True) -> Dict[str, int]:
        """
        Simulate a quantum circuit and return measurement counts.
        
        Args:
            circuit: Quantum circuit to simulate
            shots: Number of simulation shots
            add_measurements: Whether to add measurement gates
            
        Returns:
            Dictionary of measurement outcomes and their counts
        """
        sim_circuit = circuit.copy()
        
        if add_measurements:
            # Add classical register and measurements if not present
            if sim_circuit.num_clbits == 0:
                sim_circuit.add_register(ClassicalRegister(sim_circuit.num_qubits))
            sim_circuit.measure_all()
        
        # Use fake backend for realistic noise simulation
        fake_backend = FakeVigo()
        transpiled_circuit = transpile(sim_circuit, fake_backend)
        
        # Run simulation
        job = self.simulator.run(transpiled_circuit, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        return counts
    
    def analyze_trojan_impact(self, 
                            original_circuit: QuantumCircuit,
                            trojan_circuit: QuantumCircuit,
                            shots: int = 1000) -> Dict[str, float]:
        """
        Analyze the impact of Trojan insertion on circuit behavior.
        
        Args:
            original_circuit: Original quantum circuit
            trojan_circuit: Circuit with Trojan inserted
            shots: Number of simulation shots
            
        Returns:
            Dictionary containing analysis metrics
        """
        # Simulate both circuits
        original_counts = self.simulate_circuit(original_circuit, shots)
        trojan_counts = self.simulate_circuit(trojan_circuit, shots)
        
        # Calculate metrics
        tvd = self.calculate_total_variation_distance(original_counts, trojan_counts, shots)
        
        # Calculate circuit parameters
        original_depth = original_circuit.depth()
        trojan_depth = trojan_circuit.depth()
        
        original_gate_count = len(original_circuit.data)
        trojan_gate_count = len(trojan_circuit.data)
        
        analysis = {
            'tvd': tvd,
            'original_depth': original_depth,
            'trojan_depth': trojan_depth,
            'depth_increase': trojan_depth - original_depth,
            'original_gates': original_gate_count,
            'trojan_gates': trojan_gate_count,
            'gate_increase': trojan_gate_count - original_gate_count,
            'gate_increase_percent': ((trojan_gate_count - original_gate_count) / original_gate_count) * 100,
            'original_counts': original_counts,
            'trojan_counts': trojan_counts
        }
        
        return analysis

def create_benchmark_circuits() -> Dict[str, QuantumCircuit]:
    """
    Create a set of benchmark quantum circuits similar to RevLib benchmarks.
    
    Returns:
        Dictionary of benchmark circuits
    """
    circuits = {}
    
    # 1-bit Adder circuit
    adder = QuantumCircuit(3, name="1-bit_adder")
    adder.ccx(0, 1, 2)  # Toffoli gate for sum
    circuits["1-bit_adder"] = adder
    
    # Simple ALU circuit
    alu = QuantumCircuit(4, name="mini_ALU")
    alu.h(0)
    alu.cx(0, 1)
    alu.cx(1, 2)
    alu.ccx(0, 1, 3)
    circuits["mini_ALU"] = alu
    
    # Quantum Fourier Transform (4 qubits)
    qft = QuantumCircuit(4, name="QFT_4")
    # Simplified QFT implementation
    qft.h(0)
    qft.cp(np.pi/2, 0, 1)
    qft.cp(np.pi/4, 0, 2)
    qft.cp(np.pi/8, 0, 3)
    qft.h(1)
    qft.cp(np.pi/2, 1, 2)
    qft.cp(np.pi/4, 1, 3)
    qft.h(2)
    qft.cp(np.pi/2, 2, 3)
    qft.h(3)
    circuits["QFT_4"] = qft
    
    # Bell State circuit
    bell = QuantumCircuit(2, name="Bell_State")
    bell.h(0)
    bell.cx(0, 1)
    circuits["Bell_State"] = bell
    
    # GHZ State circuit
    ghz = QuantumCircuit(3, name="GHZ_State")
    ghz.h(0)
    ghz.cx(0, 1)
    ghz.cx(0, 2)
    circuits["GHZ_State"] = ghz
    
    # Random circuit for testing
    random_circuit = QuantumCircuit(5, name="Random_Circuit")
    random_circuit.h(0)
    random_circuit.cx(0, 2)
    random_circuit.ry(np.pi/4, 1)
    random_circuit.cx(1, 3)
    random_circuit.rz(np.pi/3, 4)
    random_circuit.cx(2, 4)
    circuits["Random_Circuit"] = random_circuit
    
    return circuits

# Example usage and testing
if __name__ == "__main__":
    # Create Trojan inserter
    trojan_inserter = QuantumTrojanInserter(control_qubit_index=0)
    
    # Create benchmark circuits
    benchmark_circuits = create_benchmark_circuits()
    
    # Test on Bell state circuit
    bell_circuit = benchmark_circuits["Bell_State"]
    print("Original Bell Circuit:")
    print(bell_circuit.draw())
    
    # Insert Trojan
    trojan_bell = trojan_inserter.insert_controlled_trojan(
        bell_circuit, 
        gate_limit=3, 
        activation_probability=0.5
    )
    
    print("\nTrojan-infected Bell Circuit:")
    print(trojan_bell.draw())
    
    # Analyze impact
    analysis = trojan_inserter.analyze_trojan_impact(bell_circuit, trojan_bell)
    print(f"\nAnalysis Results:")
    print(f"TVD: {analysis['tvd']:.3f}")
    print(f"Depth increase: {analysis['depth_increase']}")
    print(f"Gate increase: {analysis['gate_increase']} ({analysis['gate_increase_percent']:.1f}%)")