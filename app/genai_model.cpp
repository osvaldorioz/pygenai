#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>  // Para funciones de activación como tanh

//c++ -O3 -Wall -shared -std=c++20 -fPIC $(python3.12 -m pybind11 --includes) genai_model.cpp -o genai_model$(python3.12-config --extension-suffix)


namespace py = pybind11;

// Una clase simple que implementa una red neuronal feed-forward para generación
class GenerativeModel {
public:
    GenerativeModel(size_t input_size, size_t output_size)
        : input_size(input_size), output_size(output_size) {
        weights.resize(input_size * output_size, 0.5);  // Inicializamos pesos con 0.5
        biases.resize(output_size, 0.1);                // Inicializamos biases
    }

    // Función de activación
    double activation(double x) {
        return std::tanh(x);  // Usamos la tangente hiperbólica como función de activación
    }

    // Forward: Toma un vector de entrada y genera una salida
    std::vector<double> generate(const std::vector<double>& input) {
        if (input.size() != input_size) {
            throw std::runtime_error("El tamaño de la entrada no coincide.");
        }

        std::vector<double> output(output_size, 0.0);

        // Realizamos una multiplicación básica entre pesos y entrada
        for (size_t i = 0; i < output_size; ++i) {
            for (size_t j = 0; j < input_size; ++j) {
                output[i] += input[j] * weights[i * input_size + j];
            }
            output[i] += biases[i];  // Añadimos el bias
            output[i] = activation(output[i]);  // Aplicamos la activación
        }

        return output;
    }

private:
    size_t input_size;
    size_t output_size;
    std::vector<double> weights;
    std::vector<double> biases;
};

// Código de binding con PyBind11
PYBIND11_MODULE(genai_model, m) {
    py::class_<GenerativeModel>(m, "GenerativeModel")
        .def(py::init<size_t, size_t>())  // Constructor: input_size, output_size
        .def("generate", &GenerativeModel::generate);  // Método generate()
}
