# **Gradient Validation Strategy: Finite Differences**

To ensure the C implementation of backpropagation is correct, we will use the Central Difference method.

## **1\. Mathematical Formulation**

For any scalar parameter $\\theta$ in the model (e.g., a single weight in $W1$), the numerical gradient is:

$$\\frac{\\partial J}{\\partial \\theta} \\approx \\frac{J(\\theta \+ \\epsilon) \- J(\\theta \- \\epsilon)}{2\\epsilon}$$  
Where:

* $J$ is the total negative log-likelihood.  
* $\\epsilon$ is a small perturbation (e.g., $1e-4$).

## **2\. Implementation Logic**

### **Step A: Analytical Gradient**

1. Run maf\_forward\_train on a batch of data.  
2. Run maf\_backward.  
3. Store the resulting pointer to the specific gradient value (e.g., grad\_ana \= layer-\>grads.W1\[i\]).

### **Step B: Numerical Gradient**

1. Save the original value of the parameter: orig \= layer-\>weights.W1\[i\].  
2. **Perturb Up:**  
   * layer-\>weights.W1\[i\] \= orig \+ epsilon  
   * loss\_plus \= maf\_log\_prob(...) (Standard forward pass, no cache needed).  
3. **Perturb Down:**  
   * layer-\>weights.W1\[i\] \= orig \- epsilon  
   * loss\_minus \= maf\_log\_prob(...)  
4. **Restore:**  
   * layer-\>weights.W1\[i\] \= orig  
5. **Compute:**  
   * grad\_num \= (loss\_plus \- loss\_minus) / (2.0 \* epsilon)

### **Step C: Comparison**

Calculate Relative Error:

float numerator \= fabsf(grad\_num \- grad\_ana);  
float denominator \= fabsf(grad\_num) \+ fabsf(grad\_ana) \+ 1e-8f;  
float rel\_error \= numerator / denominator;

if (rel\_error \> 1e-4) {  
    // FAIL: Print debug info  
}

## **3\. Test Coverage**

The test harness should iterate through:

1. One random weight from $W1y$ (Masked input weights).  
2. One random weight from $W1c$ (Context weights).  
3. One random bias from $b1$.  
4. One random weight from $W2$ (Output weights).  
5. One random bias from $b2$.  
6. Repeat for First Layer, Middle Layer, and Last Layer.