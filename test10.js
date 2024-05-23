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

class GRU {
    constructor(inputSize, hiddenSize) {
        this.resetGateWeights = new Matrix(hiddenSize, inputSize + hiddenSize);
        this.updateGateWeights = new Matrix(hiddenSize, inputSize + hiddenSize);
        this.candidateWeights = new Matrix(hiddenSize, inputSize + hiddenSize);

        // Glorot initialization for weights
        const glorotFactor = Math.sqrt(2 / (inputSize + hiddenSize));
        this.resetGateWeights.randomize(-glorotFactor, glorotFactor);
        this.updateGateWeights.randomize(-glorotFactor, glorotFactor);
        this.candidateWeights.randomize(-glorotFactor, glorotFactor);

        this.resetGateBias = new Matrix(hiddenSize, 1);
        this.updateGateBias = new Matrix(hiddenSize, 1);
        this.candidateBias = new Matrix(hiddenSize, 1);

        this.resetGateBias.randomize(); // Small random bias initialization
        this.updateGateBias.randomize();
        this.candidateBias.randomize();

        this.hiddenSize = hiddenSize;
    }

    sigmoid(x) {
        return 1 / (1 + Math.exp(-x));
    }

    tanh(x) {
        return Math.tanh(x);
    }

    forward(input, prevHiddenState) {
       
        const combined =  Matrix.concatenateVertical(input, prevHiddenState);
        console.log(combined)
        const resetGate = this.sigmoid(Matrix.dot(this.resetGateWeights, combined).add(this.resetGateBias));
        const updateGate = this.sigmoid(Matrix.dot(this.updateGateWeights, combined).add(this.updateGateBias));

        const candidate = this.tanh(Matrix.dot(this.candidateWeights, combined).add(this.candidateBias));

        const newHiddenState = Matrix.multiply(updateGate, prevHiddenState)
            .add(Matrix.multiply(Matrix.subtract(Matrix.ones(updateGate.rows, updateGate.cols), updateGate), candidate))
            .add(Matrix.multiply(resetGate, candidate));

        return newHiddenState;
    }

    train(inputsSequence, targetsSequence, learningRate) {
        let prevHiddenState = new Matrix(this.hiddenSize, 1); // Initial hidden state is zero
        let totalError = 0;

        for (let t = 0; t < inputsSequence.length; t++) {
            const input = inputsSequence[t];
            const target = targetsSequence[t];

            // Forward pass
            prevHiddenState = this.forward(input, prevHiddenState);

            // Compute error
            const error = Matrix.subtract(target, prevHiddenState);
            totalError += error.map((x) => x * x).sum(); // Sum of squared errors

            // Backward pass (backpropagation through time)
            // This is a simplified version and may not work perfectly
            // In a real-world scenario, you would need to unroll the computation graph and calculate gradients accordingly

            // Compute gradients
            const gradients = error.map(this.tanh.dfunc);
            gradients.multiply(error);
            gradients.scalarMultiply(learningRate);

            // Update weights and biases
            // This is a simplified version and may not work perfectly
            // In a real-world scenario, you would need to calculate gradients for each gate and the candidate values separately
            this.candidateWeights.add(Matrix.dot(gradients, Matrix.transpose(input)));
            this.candidateBias.add(gradients);
        }

        // Compute and return Mean Squared Error (MSE)
        const mse = totalError / targetsSequence.length; // Mean Squared Error

        return mse;
    }
}

// Assuming you have a Matrix class and it has all the methods used in the GRU class

// Create a new GRU with 10 input units and 5 hidden units
const gru = new GRU(10, 5);

// Define a sequence of inputs and targets
const inputsSequence = [
    new Matrix([[1], [0], [0], [0], [0], [0], [0], [0], [0], [0]]),
    new Matrix([[0], [1], [0], [0], [0], [0], [0], [0], [0], [0]]),
  
];

const targetsSequence = [
    new Matrix([[0], [0], [0], [0], [0], [1]]),
    new Matrix([[0], [0], [0], [0], [1], [0]]),
    
];

// Train the GRU with a learning rate of 0.1
const mse = gru.train(inputsSequence, targetsSequence, 0.1);

console.log(`Mean Squared Error: ${mse}`);
