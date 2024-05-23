const axios = require('axios');







function swishActivation(inputArray) {
    // Define the Swish activation function
    function swish(x) {
        return x * (1 / (1 + Math.exp(-x)));
    }

    // Apply the Swish function to each element of the input array
    const resultArray = inputArray.map(swish);
    
    return resultArray;
}

function geluActivation(inputArray) {
    // Define the GELU activation function
    function gelu(x) {
        return 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    }

    // Apply the GELU function to each element of the input array
    const resultArray = inputArray.map(gelu);
    
    return resultArray;
}

function tanhActivation(inputArray) {
    // Apply the Tanh function to each element of the input array
    const resultArray = inputArray.map(Math.tanh);
    
    return resultArray;
}

function sigmoidActivation(inputArray) {
    // Define the Sigmoid activation function
    function sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    // Apply the Sigmoid function to each element of the input array
    const resultArray = inputArray.map(sigmoid);
    
    return resultArray;
}


class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows)
            .fill()
            .map(() => Array(this.cols).fill(0));
    }

    randomize() {
        this.data = this.data.map((row) => row.map(() => Math.random() * 2 - 1)); // Random numbers between -1 and 1
    }

    static fromArray(arr) {
        return new Matrix(arr.length, 1).map((_, i) => arr[i]);
    }

    toArray() {
        let arr = [];
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }
        return arr;
    }

    map(func) {
        this.data = this.data.map((row, i) => row.map((val, j) => func(val, i, j)));
        return this;
    }

    static map(matrix, func) {
        return new Matrix(matrix.rows, matrix.cols).map((_, i, j) => func(matrix.data[i][j], i, j));
    }

    static transpose(matrix) {
        return new Matrix(matrix.cols, matrix.rows).map((_, i, j) => matrix.data[j][i]);
    }

    static elementWiseProduct(a, b) {
        const rows = a.rows;
        const cols = a.cols;

        if (b.rows !== rows || b.cols !== cols) {
            throw new Error("Matrices dimensions don't match for element-wise multiplication");
        }

        const result = new Matrix(rows, cols);

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                result.data[i][j] = a.data[i][j] * b.data[i][j];
            }
        }

        return result;
    }

    static dot(a, b) {
        if (a.cols !== b.rows) {
            console.error('Columns of A must match rows of B for matrix multiplication.');
            return;
        }

        return new Matrix(a.rows, b.cols).map((_, i, j) => {
            // console.log(`Calculating element (${i}, ${j})`);
            let sum = 0;
            for (let k = 0; k < a.cols; k++) {
                //  console.log(`   Adding ${a.data[i][k]} * ${b.data[k][j]}`);
                sum += a.data[i][k] * b.data[k][j];
            }
            return sum;
        });
    }

    static subtract(a, b) {
        return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] - b.data[i][j]);
    }

    scalarMultiply(matrix, scalar) {
        return new Matrix(matrix.rows, matrix.cols).map((val) => val * scalar);
    }
    static scalarMultiply(matrix, scalar) {
        return new Matrix(matrix.rows, matrix.cols).map((val) => val * scalar);
    }

    sum() {
        let sum = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }
        return sum;
    }

    add(matrix) {
        if (matrix instanceof Matrix) {
            if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            this.data = this.data.map((row, i) => row.map((val, j) => val + matrix.data[i][j]));
        } else {
            this.data = this.data.map((row) => row.map((val) => val + matrix));
        }
        return this;
    }

    static multiply(a, b) {
        if (b instanceof Matrix) {
            // Hadamard product
            return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] * b.data[i][j]);
        } else {
            // Scalar product
            return new Matrix(a.rows, a.cols).map((_, i, j) => a.data[i][j] * b);
        }
    }
    clone() {
        const newMatrix = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                newMatrix.data[i][j] = this.data[i][j];
            }
        }
        return newMatrix;
    }

    subtract(matrix) {
        if (matrix instanceof Matrix) {
            if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            this.data = this.data.map((row, i) => row.map((val, j) => val - matrix.data[i][j]));
        } else {
            this.data = this.data.map((row) => row.map((val) => val - matrix));
        }
        return this;
    }

    multiply(matrix) {
        if (matrix instanceof Matrix) {
            if (this.rows !== matrix.rows || this.cols !== matrix.cols) {
                console.error('Columns and Rows of A must match Columns and Rows of B.');
                return;
            }
            this.data = this.data.map((row, i) => row.map((val, j) => val * matrix.data[i][j]));
        } else {
            this.data = this.data.map((row) => row.map((val) => val * matrix));
        }
        return this;
    }
    static concatenateVertical(a, b) {
        if (a.cols !== b.cols) {
            console.error('Number of columns must match for vertical concatenation.');
            return;
        }

        const resultRows = a.rows + b.rows;
        const resultCols = a.cols;
        const resultData = [];

        for (let i = 0; i < a.rows; i++) {
            resultData.push([...a.data[i]]);
        }

        for (let i = 0; i < b.rows; i++) {
            const row = [];
            for (let j = 0; j < b.cols; j++) {
                row.push(b.data[i][j]);
            }
            resultData.push(row);
        }
        

        const result = new Matrix(resultRows, resultCols);
        result.data = resultData;
        return result;
    }

    static multiplyElementWise(a, b) {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            console.error('Dimensions must match for element-wise multiplication.');
            return;
        }

        const result = new Matrix(a.rows, a.cols);
        result.data = a.data.map((row, i) => row.map((val, j) => val * b.data[i][j]));
        return result;
    }
}

class Layer {
    constructor(inputSize, outputSize, activation) {
        this.weights = new Matrix(outputSize, inputSize);
        // Glorot initialization for weights
        const glorotFactor = Math.sqrt(2 / (inputSize + outputSize));
        this.weights.randomize(-glorotFactor, glorotFactor);
        this.bias = new Matrix(outputSize, 1);
        this.bias.randomize(); // Small random bias initialization
        this.activation = activation;
    }

    forward(input) {
        this.input = input; 
        this.weightedSum = Matrix.dot(this.weights, input).add(this.bias);

        this.output = this.weightedSum.map(this.activation.func);

        return this.output;
    }


    train(input, target, learningRate) {
        // Forward pass
       
        const output = this.forward(input);

        // Compute error
        let error;
        error = Matrix.subtract(target, output);

        // Compute gradients
        let gradients;
        gradients = this.weightedSum.map(this.activation.dfunc);
        gradients.multiply(error);
        gradients.scalarMultiply(learningRate);

        // Compute weight deltas
        const deltas = Matrix.dot(gradients, Matrix.transpose(input));

        // Update weights and biases
        this.weights.add(deltas);
        this.bias.add(gradients);

        // Compute and return Mean Squared Error (MSE)
        const sumSquaredErrors = error.map((x) => x * x).sum(); // Sum of squared errors
        const mse = sumSquaredErrors / error.rows; // Mean Squared Error

        return mse;
    }
}


const sigmoid = {
    func: (x) => 1 / (1 + Math.exp(-x)),
    dfunc: (y) => y * (1 - y), // derivative of sigmoid
};

const tanh = {
    func: (x) => Math.tanh(x),
    dfunc: (y) => 1 - y * y, // derivative of tanh
};

const leakyRelu = {
    func: (x) => (x > 0 ? x : 0.01 * x), // Leaky ReLU with alpha = 0.01
    dfunc: (y) => (y > 0 ? 1 : 0.01), // derivative of Leaky ReLU
};
const swish = {
    func: (x) => x / (1 + Math.exp(-x)),
    dfunc: (y, x) => y + (1 - y / (1 + Math.exp(-x))),
};

const relu = {
    func: (x) => Math.max(0, x),
    dfunc: (y) => (y > 0 ? 1 : 0), // derivative of ReLU
};


const softplus = {
    func: (x) => Math.log(1 + Math.exp(x)),
    dfunc: (y) => 1 / (1 + Math.exp(-y)), // derivative of softplus
};
const elu = {
    alpha: 0.01, // or any small positive value
    func: (x) => (x > 0 ? x : elu.alpha * (Math.exp(x) - 1)),
    dfunc: (y) => (y > 0 ? 1 : y + elu.alpha), // derivative of ELU
};
const gelu = {
    func: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)))),
    dfunc: (x) => {
        const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
        return 0.5 * (1 + Math.pow(cdf, 2) * (1 - Math.sqrt(2 / Math.PI) * (x + 0.044715 * Math.pow(x, 3)) * Math.exp(-(Math.pow(x, 2) / 2))));
    },
};

const linear = {
    func: (x) => x,
    dfunc: () => 1, // derivative of linear function is always 1
};
class GRU {
    constructor(inputSize, hiddenSize) {
        this.resetGate = new Layer(inputSize + hiddenSize, hiddenSize, sigmoid);
        this.updateGate = new Layer(inputSize + hiddenSize, hiddenSize, sigmoid);
        this.candidate = new Layer(inputSize + hiddenSize, hiddenSize, tanh);
    }

    forward(input, prevHidden) {
       
        const combined = Matrix.concatenateVertical(input, prevHidden);

        const resetGate = this.resetGate.forward(combined);
        const updateGate = this.updateGate.forward(combined);
        const candidate = this.candidate.forward(combined);

        const resetCandidate = Matrix.multiply(resetGate, candidate);
        const newHidden = Matrix.multiply(updateGate, prevHidden).add(resetCandidate);

        return newHidden;
    }

    train(inputs, targets, learningRate) {
        const inputsLength = inputs.length;
        const outputs = [];
        let prevHidden = new Matrix(this.hiddenSize, 1);
    
        // Forward pass
        for (let i = 0; i < inputsLength; i++) {
            prevHidden = this.forward(inputs[i], prevHidden);
            outputs.push(prevHidden);
        }
    
        // Backward pass through time
        let gradients = [];
        for (let t = inputsLength - 1; t >= 0; t--) {
            const error = Matrix.subtract(targets[t], outputs[t]);
            const combined = Matrix.concatVertically(inputs[t], prevHidden);
    
            // Calculate gradients for each gate and candidate
            // This is a simplified version, you might need to adjust this based on your specific use case
            const candidateGradients = this.candidate.train(combined, error, learningRate);
            const updateGateGradients = this.updateGate.train(combined, error, learningRate);
            const resetGateGradients = this.resetGate.train(combined, error, learningRate);
    
            // Combine gradients for prevHidden
            const nextPrevHiddenGradients = Matrix.add(candidateGradients, resetGateGradients);
            const prevHiddenGradients = Matrix.add(nextPrevHiddenGradients, updateGateGradients);
    
            gradients.unshift(prevHiddenGradients);
        }
    
        // Update weights and biases using gradients
        // This is a simplified version, you might need to adjust this based on your specific use case
        for (let t = 0; t < inputsLength; t++) {
            this.candidate.weights.subtract(gradients[t]);
            this.updateGate.weights.subtract(gradients[t]);
            this.resetGate.weights.subtract(gradients[t]);
        }
    }    
}


// Assume Matrix and Layer classes are defined and working correctly

// Initialize a GRU with input size 5 and hidden size 10
const gru = new GRU(3, 10);

// Assume some input and target data
const inputs = Matrix.fromArray([0.1,0.2,0.3]);
const targets = Matrix.fromArray([0.2,0.3,0.4]);

// Assume a learning rate
const learningRate = 0.01;

// Train the GRU
for(i=0;i<100000;i++){
    gru.train(inputs, targets, learningRate);
}

// After training, you can use the GRU to make predictions
const someInput = new Matrix(3, 1); // some input data
let hiddenState = new Matrix(10, 1); // initialize hidden state
sequenceLength = 3;

// Loop over a sequence of inputs
for (let i = 0; i < sequenceLength; i++) {
    hiddenState = gru.forward(someInput, hiddenState);
    console.log("hS",hiddenState)
}
