#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <memory>

class Tensor : public std::enable_shared_from_this<Tensor> {
public:
    double value;  // The value of the tensor
    double grad;   // The gradient of the tensor
    std::vector<std::shared_ptr<Tensor>> parents; // Parent tensors in the computation graph
    std::vector<double> local_grads; // Local gradients with respect to parents

    // Constructor
    Tensor(double value) : value(value), grad(0) {}

    // Sigmoid activation function
    std::shared_ptr<Tensor> sigmoid() {
        double s = 1.0 / (1.0 + exp(-this->value));
        auto result = std::make_shared<Tensor>(s);
        result->parents.push_back(shared_from_this());
        result->local_grads.push_back(s * (1 - s));
        return result;
    }

    // Backward pass for automatic differentiation
    void backward(double grad=1.0) {
        this->grad += grad;
        for(size_t i = 0; i < parents.size(); ++i) {
            parents[i]->backward(grad * local_grads[i]);
        }
    }
};

// Overload addition operator
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    auto result = std::make_shared<Tensor>(a->value + b->value);
    result->parents.push_back(a);
    result->local_grads.push_back(1.0);
    result->parents.push_back(b);
    result->local_grads.push_back(1.0);
    return result;
}

// Overload multiplication operator
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
    auto result = std::make_shared<Tensor>(a->value * b->value);
    result->parents.push_back(a);
    result->local_grads.push_back(b->value);
    result->parents.push_back(b);
    result->local_grads.push_back(a->value);
    return result;
}

int main() {
    // Training data for XOR problem
    std::vector<std::vector<double>> inputs = {
        {0, 0}, {0, 1}, {1, 0}, {1, 1}
    };
    std::vector<double> targets = {0, 1, 1, 0};
    const double lr = 0.5; // Learning rate

    // Initialize weights and biases as shared pointers to tensors with random values
    auto w1 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto w2 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto w3 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto w4 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto b1 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto b2 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto w5 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto w6 = std::make_shared<Tensor>(rand() / double(RAND_MAX));
    auto b3 = std::make_shared<Tensor>(rand() / double(RAND_MAX));

    // Training loop
    for(int epoch = 0; epoch < 10000; ++epoch) {
        for(size_t i = 0; i < inputs.size(); ++i) {
            // Reset gradients before each sample
            w1->grad = w2->grad = w3->grad = w4->grad = 0;
            b1->grad = b2->grad = 0;
            w5->grad = w6->grad = 0;
            b3->grad = 0;

            // Forward pass
            auto x1 = std::make_shared<Tensor>(inputs[i][0]);
            auto x2 = std::make_shared<Tensor>(inputs[i][1]);

            auto h1 = x1 * w1 + x2 * w2 + b1;
            h1 = h1->sigmoid();

            auto h2 = x1 * w3 + x2 * w4 + b2;
            h2 = h2->sigmoid();

            auto o1 = h1 * w5 + h2 * w6 + b3;
            o1 = o1->sigmoid();

            // Compute loss (mean squared error)
            double y = targets[i];
            double loss = 0.5 * (o1->value - y) * (o1->value - y);

            // Backward pass
            o1->backward(o1->value - y);

            // Update weights and biases using gradients
            w1->value -= lr * w1->grad;
            w2->value -= lr * w2->grad;
            w3->value -= lr * w3->grad;
            w4->value -= lr * w4->grad;
            w5->value -= lr * w5->grad;
            w6->value -= lr * w6->grad;
            b1->value -= lr * b1->grad;
            b2->value -= lr * b2->grad;
            b3->value -= lr * b3->grad;
        }
    }

    // Inference after training
    for(size_t i = 0; i < inputs.size(); ++i) {
        auto x1 = std::make_shared<Tensor>(inputs[i][0]);
        auto x2 = std::make_shared<Tensor>(inputs[i][1]);

        auto h1 = x1 * w1 + x2 * w2 + b1;
        h1 = h1->sigmoid();

        auto h2 = x1 * w3 + x2 * w4 + b2;
        h2 = h2->sigmoid();

        auto o1 = h1 * w5 + h2 * w6 + b3;
        o1 = o1->sigmoid();

        std::cout << "Input: " << inputs[i][0] << " " << inputs[i][1]
                  << " Output: " << o1->value << std::endl;
    }

    return 0;
}
