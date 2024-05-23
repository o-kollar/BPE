

class Matrix {
    constructor(rows, cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = Array(this.rows)
            .fill()
            .map(() => Array(this.cols).fill(0));
    }

    submatrix(startRow, startCol, endRow, endCol) {
        let subData = [];
        for (let i = startRow; i <= endRow; i++) {
            let row = [];
            for (let j = startCol; j <= endCol; j++) {
                row.push(this.data[i][j]);
            }
            subData.push(row);
        }
        return new Matrix(subData.length, subData[0].length).map((_, i, j) => subData[i][j]);
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
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;

        this.Wz = new Matrix(hiddenSize, inputSize + hiddenSize);
        this.Wz.randomize()
        this.Wr = new Matrix(hiddenSize, inputSize + hiddenSize);
        this.Wr.randomize()
        this.Wh = new Matrix(hiddenSize, inputSize + hiddenSize);
        this.Wh.randomize()

        this.bh = Matrix.fromArray([1,1,1,1,1]);
        this.bz = Matrix.fromArray([1,1,1,1,1]);
        this.br = Matrix.fromArray([1,1,1,1,1]);
    }

    tanhDerivative(x) {
        return x.map((val) => 1 - Math.pow(Math.tanh(val), 2));
    }

    forward(input) {
        let hidden = Matrix.fromArray([1,1,1,1,1]);
        let states = [];

        for (let i = 0; i < input.rows; i++) {
            let x = input.data[i][0];
            let r = this.sigmoid(Matrix.dot(this.Wr ,Matrix.concatenateVertical(Matrix.fromArray([x]), hidden)).add(this.br));
            let z = this.sigmoid(Matrix.dot(this.Wz,Matrix.concatenateVertical(Matrix.fromArray([x]), hidden)).add(this.bz));
            let hPrev = hidden;
            console.log(Matrix.fromArray([1,1,1,1,1]).subtract(z),hidden)
            hidden = (Matrix.multiplyElementWise(Matrix.fromArray([1,1,1,1,1]).subtract(z),hidden))
            hidden.add(Matrix.multiplyElementWise(z,this.tanh(Matrix.dot(this.Wh,Matrix.concatenateVertical(Matrix.fromArray([x]), Matrix.multiplyElementWise(r,hPrev))).add(this.bh))));
            states.push(hidden.clone());
        }

        return states;
    }

    sigmoidDerivative(x) {
        let sigmoid = this.sigmoid(x);
        return x.map((_, i, j) => sigmoid.data[i][j] * (1 - sigmoid.data[i][j]));
    }

    sigmoid(x) {
        return x.map((val) => 1 / (1 + Math.exp(-val)));
    }

    tanh(x) {
        return x.map((val) => Math.tanh(val));
    }
}

const sigmoid = {
    func: (x) => 1 / (1 + Math.exp(-x)),
    dfunc: (y) => y * (1 - y), // derivative of sigmoid
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

const tanh = {
    func: (x) => Math.tanh(x),
    dfunc: (y) => 1 - y * y, // derivative of tanh
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

class BPTT {
    constructor(gru, target) {
        this.gru = gru;
        this.target = target;
    }

    train(input, learningRate) {
        let states = this.gru.forward(input);
        let errors = [];
        let dWh = new Matrix(this.gru.Wh.rows, this.gru.Wh.cols);
        let dWz = new Matrix(this.gru.Wz.rows, this.gru.Wz.cols);
        let dWr = new Matrix(this.gru.Wr.rows, this.gru.Wr.cols);
        let dbh = new Matrix(this.gru.bh.rows, this.gru.bh.cols);
        let dbz = new Matrix(this.gru.bz.rows, this.gru.bz.cols);
        let dbr = new Matrix(this.gru.br.rows, this.gru.br.cols);

        for (let t = states.length - 1; t >= 0; t--) {
            let state = states[t];
            let error = state.subtract(this.target);
            errors.unshift(error);

            let dtanh = Matrix.dot(Matrix.transpose(this.gru.Wh),Matrix.multiplyElementWise(error,this.gru.tanhDerivative(state)));
            let dz = Matrix.multiplyElementWise(Matrix.multiplyElementWise(error,state),this.gru.sigmoidDerivative(Matrix.dot(this.gru.Wz,Matrix.concatenateVertical(input.submatrix(t, 1, t, input.cols), states[t])).add(this.gru.bz)));
            let dr = Matrix.multiplyElementWise(Matrix.multiplyElementWise(Matrix.dot(this.gru.Wh,Matrix.concatenateVertical(input.submatrix(t, 1, t, input.cols), states[t])),Matrix.multiplyElementWise(error,state)),this.gru.sigmoidDerivative(Matrix.dot(this.gru.Wr,Matrix.concatenateVertical(input.submatrix(t, 1, t, input.cols), states[t])).add(this.gru.br)));


            dWh = Matrix.transpose(dWh.add(Matrix.dot(dtanh,Matrix.concatenateVertical(input.submatrix(t, 1, t, input.cols), states[t]))));
            dWz = Matrix.transpose(dWz.add(Matrix.dot(dz,Matrix.concatenateVertical(input.submatrix(t, 1, t, input.cols), states[t]))));
            dWr = Matrix.transpose(dWr.add(Matrix.dot(dr,Matrix.concatenateVertical(input.submatrix(t, 1, t, input.cols), states[t]))));
            

            console.log("dtang",dtanh)
            dbh = dbh.add(dtanh);
            dbz = dbz.add(dz);
            dbr = dbr.add(dr);
        }

        this.gru.Wh = this.gru.Wh.subtract(dWh.scalarMultiply(learningRate));
        this.gru.Wz = this.gru.Wz.subtract(dWz.scalarMultiply(learningRate));
        this.gru.Wr = this.gru.Wr.subtract(dWr.scalarMultiply(learningRate));
        this.gru.bh = this.gru.bh.subtract(dbh.scalarMultiply(learningRate));
        this.gru.bz = this.gru.bz.subtract(dbz.scalarMultiply(learningRate));
        this.gru.br = this.gru.br.subtract(dbr.scalarMultiply(learningRate));
    }
}

// Example usage
let inputSize = 1;
let hiddenSize = 5;
let gru = new GRU(inputSize, hiddenSize);
let bptt = new BPTT(gru, Matrix.fromArray([0.1, 0.2, 0.3, 0.4, 0.5]));
let input = Matrix.fromArray([0.1, 0.2, 0.3, 0.4, 0.5]);
bptt.train(input, 0.1);

