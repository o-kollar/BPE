const tf = require('@tensorflow/tfjs');

const sequence = "Hello World!";

// Define the RNN model
const model = tf.sequential();
model.add(tf.layers.lstm({units: 128, inputShape: [1, sequence.length]}));
model.add(tf.layers.dense({units: sequence.length, activation: 'softmax'}));

// Compile the model
model.compile({optimizer: 'adam', loss: 'categoricalCrossentropy', metrics: ['accuracy']});

// Generate training data
const data = [];
const labels = [];
for (let i = 0; i < sequence.length - 1; i++) {
    data.push(sequence.substring(i, i + 1));
    labels.push(sequence.substring(i + 1, i + 2));
}

// One-hot encode the data
const encodedData = tf.oneHot(tf.tensor1d(data.map(c => sequence.indexOf(c))), sequence.length);
const encodedLabels = tf.oneHot(tf.tensor1d(labels.map(c => sequence.indexOf(c))), sequence.length);

// Train the model
async function train(){
await model.fit(encodedData, encodedLabels, {epochs: 100});
}
// Generate a new sequence
const seed = "H";
let generated = seed;
for (let i = 0; i < sequence.length - 1; i++) {
    const input = tf.oneHot(tf.tensor1d([sequence.indexOf(generated.substring(i, i + 1))]), sequence.length);
    const output = model.predict(input);
    const index = output.argMax(-1).dataSync()[0];
    generated += sequence.substring(index, index + 1);
}
console.log(`Original sequence: ${sequence}`);
console.log(`Generated sequence: ${generated}`);

train()