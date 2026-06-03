def derivative(x):
    # Return the derivative of f(x) = x^2 -4x + 4
    return 2 * x - 4

def run_gradident_descent(current_x, learning_rate, iterations):

    print(f"Starting gradient descent at x = {current_x}")
    
    for i in range(iterations):
        slope = derivative(current_x)
        current_x = current_x - (learning_rate * slope)
        if (i + 1) % 5 == 0:
            print(f"Iteration {i+1}: x = {current_x:.4f}, slope = {slope:.4f}")

    print(f"\nOptimization finished. Found minimum at x = {current_x:.4f}")

if __name__ == "__main__":
    run_gradident_descent(5, 0.1, 50)