import matplotlib.pyplot as plt
import numpy as np
# Quantum gates (from previous code)
def h_gate(qubit, state):
    # Hadamard gate on specified qubit
    hadamard_matrix = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
    identity_matrix = np.eye(2)
    
    # Apply Hadamard gate on the specified qubit
    state = np.kron(hadamard_matrix, identity_matrix) @ state
    
    return state

def cnot_gate(control, target, state):
    # Controlled NOT gate on specified qubits
    cnot_matrix = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1],
                            [0, 0, 1, 0]])
    
    # Apply CNOT gate
    state = np.kron(np.eye(2**control), np.kron(cnot_matrix, np.eye(2**(target - control - 1)))) @ state
    
    return state

# Function to map quantum state to classical probabilities
def get_classical_probabilities(state):
    probabilities = np.abs(state) ** 2
    return probabilities / np.sum(probabilities)

# Visualize decision boundary in input space
x_values = np.linspace(0, 1, 100)
y_values = np.linspace(0, 1, 100)

decision_boundary = []

for x in x_values:
    for y in y_values:
        # Convert input to quantum state
        input_state = np.array([1, 0, 0, 0])
        input_state = h_gate(0, input_state)
        input_state = h_gate(1, input_state)

        # Apply CNOT gate based on the input
        if x == 1:
            input_state = cnot_gate(0, 1, input_state)

        # Map quantum state to classical probabilities
        probabilities = get_classical_probabilities(input_state)
        
        # Use the probability of the output state being in the |1⟩ state
        decision_boundary.append(((x, y), probabilities[2]))

# Separate x and y values for plotting
x_decision, y_decision = zip(*[point[0] for point in decision_boundary])
colors = [point[1] for point in decision_boundary]

# Plot decision boundary
plt.scatter(x_decision, y_decision, c=colors, cmap='viridis', marker='.')
plt.xlabel('Input X1')
plt.ylabel('Input X2')
plt.title('Decision Boundary Visualization')
plt.colorbar(label='Probability of Output being |1⟩')
plt.show()
