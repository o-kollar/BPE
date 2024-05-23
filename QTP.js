const fs = require('fs');
const axios = require('axios')

class Vocab {
    constructor() {
        this.vocab = [];
    }

    build(inputString, vocabSize) {
        // Step 1: Initialize vocabulary with individual characters
        const vocab = new Set(inputString.split(''));

        // Step 2: Perform Byte Pair Encoding
        while (vocab.size < vocabSize) {
            // Step 2a: Count pairs frequency
            const pairs = {};
            inputString.split('').forEach((letter, index) => {
                if (index < inputString.length - 1) {
                    const pair = inputString.slice(index, index + 2);
                    if (!pairs[pair]) {
                        pairs[pair] = 0;
                    }
                    pairs[pair]++;
                }
            });

            // Step 2b: Find the most frequent pair
            let mostFrequentPair = null;
            let maxFrequency = 0;
            for (const pair in pairs) {
                if (pairs[pair] > maxFrequency) {
                    mostFrequentPair = pair;
                    maxFrequency = pairs[pair];
                }
            }

            // Step 2c: Merge the most frequent pair
            if (mostFrequentPair) {
                vocab.add(mostFrequentPair);
                inputString = inputString.split(mostFrequentPair).join('');
            } else {
                // No more pairs to merge, break the loop
                break;
            }
        }

        // Step 3: Update vocabulary
        this.vocab = Array.from(vocab);
    }

    tokenize(str) {
        return str.split('').map((letter) => this.vocab.indexOf(letter) / (this.vocab.length - 1));
    }
     tokenizeChar(char) {
        return this.vocab.indexOf(char) / (this.vocab.length - 1);
    }

    detokenize(tokens) {
        return tokens.map((token) => this.vocab[Math.round(token * (this.vocab.length - 1))]).join('');
    }
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
    static randomNormal(rows, cols) {
        const matrix = new Matrix(rows, cols);
        for (let i = 0; i < rows; i++) {
          for (let j = 0; j < cols; j++) {
            matrix.data[i * cols + j] = Math.random() * 2 - 1; // Uniform distribution between -1 and 1
          }
        }
        return matrix;
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
        console.log("input",input)
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



class Graph {
  constructor() {
    this.vertices = {};
  }

  addVertex(vertex) {
    if (!this.vertices[vertex]) {
      this.vertices[vertex] = {};
    }
  }

  setEdge(vertex1, vertex2, reward, probability) {
    if (!this.vertices[vertex1]) {
      this.addVertex(vertex1);
    }
    if (!this.vertices[vertex2]) {
      this.addVertex(vertex2);
    }
    if (!this.vertices[vertex1][vertex2]) {
      // Initialize the edge if it doesn't exist
      this.vertices[vertex1][vertex2] = { reward, probability: 0, count: 0 };
    }
    // Update the edge
    this.vertices[vertex1][vertex2].reward = (this.vertices[vertex1][vertex2].reward * this.vertices[vertex1][vertex2].count + reward) / (this.vertices[vertex1][vertex2].count + 1);
    this.vertices[vertex1][vertex2].probability = (this.vertices[vertex1][vertex2].probability * this.vertices[vertex1][vertex2].count + probability) / (this.vertices[vertex1][vertex2].count + 1);
    this.vertices[vertex1][vertex2].count++;
  }

  load(filename) {
    const data = fs.readFileSync(filename);
    this.vertices = JSON.parse(data);
    console.log(`Graph loaded from ${filename}`);
  }

  save(filename) {
    const data = JSON.stringify(this.vertices, null, 2);
    fs.writeFileSync(filename, data);
    console.log(`Graph saved to ${filename}`);
  }

  getVertices(vertex) {
    const neighbors = this.vertices[vertex];
    if (!neighbors) {
      throw new Error('Vertex does not exist');
    }
    return Object.entries(neighbors).map(([vertex, { reward, probability }]) => ({ vertex, reward, probability }));
  }
}

const STATES = ['A', 'B', 'C','E'];
const ACTIONS = ['a', 'b'];

class QTP {
    constructor(actions) {
        this.graph = new Graph();
        this.states = [];
        this.actions = actions;
        this.actionValues = {}; // Store distribution parameters for each action
    }

    getIndex(arr, item) {
        return arr.indexOf(item);
    }

    updateTransitionProbabilities(data) {
        for (const state in data) {
            const transitions = data[state];
            const totalTransitions = Object.values(transitions).reduce((acc, transition) => acc + transition.count, 0);

            for (const nextState in transitions) {
                const transition = transitions[nextState];
                const probability = transition.count / totalTransitions;
                this.graph.setEdge(state, nextState, transition.reward, probability);
            }
        }
    }
    
    step(state) {
        this.states.push(state)
        this.graph.addVertex(state)
        this.states.forEach(point => this.graph.setEdge(state,point,0,0.5))
        const action = this.chooseAction(this.lastReward);
        const neighbors = this.graph.getVertices(state);
        const weights = neighbors.map(neighbor => neighbor.probability);
        const totalWeight = weights.reduce((total, weight) => total + weight, 0);
        const randomChoice = Math.random() * totalWeight;
        let cumulativeWeight = 0;
        let nextState = null;
        let nextStateProbability = 0;

        
        let maxScore = -Infinity;
      if(action === 'a'){
        nextState = state
      }else if(action ==='b'){
        for (const [index, neighbor] of neighbors.entries()) {
            cumulativeWeight += weights[index];
            const score = weights[index] * neighbor.reward; // Combine probability and reward
            if (score > maxScore || (score === maxScore && Math.random() < 0.5)) {
                // If score is greater than maxScore or equal and randomly chosen, update nextState
                nextState = neighbor.vertex;
                nextStateProbability = neighbor.probability;
                maxScore = score;
            }
        }
      }
        


        if (!nextState) {
            nextState = state;
            nextStateProbability = 1;
        }

        const reward = this.reward(state, action, nextState);
        this.graph.setEdge(state, nextState, reward, nextStateProbability);
        this.lastReward = reward;
        this.updateTransitionProbabilities(graph.vertices)


        switch (nextState) {
            case 'A':
                console.log('piff')
                break;
            case 'B':
            console.log('puff')
            break;
            case 'C':
            console.log('paff')
            break;
            default:
                break;
        }

        this.graph.save('graph.json')

        return { newState: nextState, reward: this.lastReward };
    }

    chooseAction(lastReward) {
        const initialEpsilon = 0.9; // Initial exploration rate
        const minEpsilon = 0.01; // Minimum exploration rate
        const epsilonDecay = 0.001; // Epsilon decay rate
        const epsilon = Math.max(minEpsilon, initialEpsilon * Math.exp(-epsilonDecay * this.totalSteps));

        this.totalSteps++; // Increment total steps

        if (Math.random() > epsilon) {
            const randomIndex = Math.floor(Math.random() * this.actions.length);
            return this.actions[randomIndex];
            
        } else {
            return lastReward === null || lastReward > 0 ? 'a' : 'b';
            
        }
    }

    reward(state, action, nextState) {
        if (nextState === 'B') return 1;
        else if (nextState === 'C') return -1;
        else return 0;
    }
}

// Usage
const graph = new Graph(); // Assume Graph implementation is provided





const qtp = new QTP(ACTIONS);

for(i=0;i<2;i++){
STATES.forEach(d => qtp.step(d))
}
    





SampleText = `In JavaScript, you can use a switch statement to perform a multiway branch based on the value of an expression. Here's the basic syntax:`
vocab = new Vocab()
vocab.build(SampleText,10);
class MarkovChain {
    constructor(numStates, layerSize, learningRate) {
      this.numStates = numStates;
      this.layer = new Layer(numStates, layerSize, sigmoid);
      this.outputLayer = new Layer(layerSize, numStates, relu);
      this.learningRate = learningRate;
      this.graph = new Graph();
    }
  
    // Train the Markov chain on a sequence of states
    train(sequence, numEpochs) {
      for (let epoch = 0; epoch < numEpochs; epoch++) {
        for (let i = 0; i < sequence.length - 1; i++) {
          const input = this.oneHotEncode(sequence[i]);
          const target = this.oneHotEncode(sequence[i + 1]);
  
          // Forward pass through the network
          const layerOutput = this.layer.forward(input);
          const output = this.outputLayer.forward(layerOutput);
  
          // Update the graph with the transition probability
          this.graph.setEdge(sequence[i], sequence[i + 1], 1, output.data[sequence[i + 1]][0]);
          this.graph.save('GGG.json')
  
          // Backward pass to update weights and biases
          const mse = this.outputLayer.train(layerOutput, target, this.learningRate);
          this.layer.train(input, layerOutput, this.learningRate);
        }
      }
    }
  
    // Generate a new state based on the current state and previous states
    generate(currentState, prevStates) {
      const input = this.getInput(currentState, prevStates);
      const layerOutput = this.layer.forward(input);
      const output = this.outputLayer.forward(layerOutput);
      
        console.log("INPUT",input)
      // Sample a new state based on the output probabilities
      const nextStateIndex = this.sampleFromProbabilities(output.data);
      return nextStateIndex;
    }
  
    // Get the input vector for the neural network based on the current state and previous states
    getInput(currentState, prevStates) {
      const input = new Matrix(this.numStates + prevStates.length, 1);
      input.data[currentState] = 1;
      for (let i = 0; i < prevStates.length; i++) {
        input.data[this.numStates + i] = prevStates[i];
      }
      return input;
    }
  
    // One-hot encode a state index as a vector
    oneHotEncode(index) {
      const vector = Matrix.fromArray(Array(this.numStates).fill(0).map((_, i) => i === index ? 1 : 0))
      return vector;
    }
  
    // Sample an index from a probability distribution
    sampleFromProbabilities(probabilities) {
      const r = Math.random();
      let cumulativeProbability = 0;
      for (let i = 0; i < probabilities.length; i++) {
        cumulativeProbability += probabilities[i];
        if (r <= cumulativeProbability) {
          return i;
        }
      }
      // Should never get here, but return a default value just in case
      return 0;
    }
  }
  
  const numStates = 6; // Number of states
  const layerSize = 8; // Number of neurons in the hidden layer const learningRate = 0.1; // Learning rate for the neural network
  
  // Define a sequence of states to train the model on const 
  sequence = [5, 1, 2, 1, 0, 3, 2, 1, 0, 2, 3, 1, 2, 0, 1];
  
  // Create a new Markov chain with a given learning rate const 
  markovChain = new MarkovChain(numStates, layerSize, 0.01);
  
  // Train the Markov chain on the sequence for 1000 epochs 
  markovChain.train(sequence, 10);
  
  // Generate a new sequence of 10 states starting from state 0 let 
  currentState = 1; const generatedSequence = [];
   for (let i = 0; i < 10; i++) { generatedSequence.push(currentState),MarkovChain}; // Use the previous 3 states as input to the neural network const prevStates = generatedSequence.slice(Math.max(generatedSequence.length - 3, 0)); currentState = markovChain.generate(currentState, prevStates); }
  
  console.log("Generated Seq",generatedSequence); // Output: e.g. [0, 1, 2, 1, 0, 2, 3, 1, 2, 0]