const fs = require('fs')
class MarkovChain {
    constructor(order) {
        this.order = order; // Order of the Markov chain
        this.chain = {}; // The Markov chain model
    }

    train(text) {
        // Clean the text and split it into words
        const words = text.toLowerCase().match(/'(?:[sdmt]|ll|ve|re)|\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4})\b|[\p{L}\p{M}]+|[\p{N}\p{Nd}]+|[^\s\p{L}\p{N}]+|\s+/gu);

        // Loop through the words to build the Markov chain model
        for (let i = 0; i < words.length - this.order; i++) {
            const state = words.slice(i, i + this.order).join(' ');
            const nextWord = words[i + this.order];
            if (!this.chain[state]) {
                this.chain[state] = [];
            }
            this.chain[state].push(nextWord);
        }
    }

    generate(length, initialSentence) {
        let result = initialSentence.toLowerCase().match(/'(?:[sdmt]|ll|ve|re)|\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4})\b|[\p{L}\p{M}]+|[\p{N}\p{Nd}]+|[^\s\p{L}\p{N}]+|\s+/gu);

        // Generate the rest of the sequence
        let currentState = result.slice(-this.order).join(' ');
        for (let i = 0; i < length - this.order; i++) {
            if (!this.chain[currentState]) {
                break;
            }
            const nextWord = this._getRandomElement(this.chain[currentState]);
            result.push(nextWord);
            currentState = result.slice(-this.order).join(' ');
        }

        return result.join(' ');
    }

    saveModel(filename) {
        const data = JSON.stringify(this.chain);
        fs.writeFileSync(filename, data);
        console.log('Model saved successfully.');
    }

    loadModel(filename) {
        try {
            const data = fs.readFileSync(filename, 'utf8');
            this.chain = JSON.parse(data);
            console.log('Model loaded successfully.');
        } catch (error) {
            console.error('Error loading model:', error);
        }
    }

    _getRandomElement(arr) {
        return arr[Math.floor(Math.random() * arr.length)];
    }
}

const order = 4; // Order of the Markov chain
const chain = new MarkovChain(order);
chain.loadModel('markov_model.json');
const initialSentence = `the most `;
console.log(chain.generate(100, initialSentence))