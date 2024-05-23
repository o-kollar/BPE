const axios = require('axios')
const fs = require('fs')
class BayesianMarkovChain {
  constructor(order, alpha = 0.5, beta = 0.5) {
    this.order = order; // Order of the Markov chain
    this.chain = {}; // The Markov chain model
    this.alpha = alpha; // Smoothing parameter for unseen states
    this.beta = beta; // Smoothing parameter for unseen next words
    this.prior = {}; // Prior distribution over transition probabilities
  }

  
    saveModel(filename) {
      const data = {
        chain: this.chain,
        prior: this._serializePrior(this.prior),
      };
      fs.writeFileSync(filename, JSON.stringify(data));
      console.log('Model saved successfully.');
    }
  
    loadModel(filename) {
      try {
        const data = fs.readFileSync(filename, 'utf8');
        const loadedData = JSON.parse(data);
        this.chain = loadedData.chain;
        this.prior = this._deserializePrior(loadedData.prior);
        console.log('Model loaded successfully.');
      } catch (error) {
        console.error('Error loading model:', error);
      }
    }
  
    _serializePrior(prior) {
      const serializedPrior = {};
      for (const state in prior) {
        serializedPrior[state] = {};
        for (const nextWord in prior[state]) {
          serializedPrior[state][nextWord] = prior[state][nextWord];
        }
      }
      return serializedPrior;
    }
  
    _deserializePrior(serializedPrior) {
      const prior = {};
      for (const state in serializedPrior) {
        prior[state] = {};
        for (const nextWord in serializedPrior[state]) {
          prior[state][nextWord] = serializedPrior[state][nextWord];
        }
      }
      return prior;
    }
  
  train(text) {
    // Clean the text and split it into words
    const words = text.toLowerCase().match(/'(?:[sdmt]|ll|ve|re)|\b(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{2,4}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{2,4})\b|[\p{L}\p{M}]+|[\p{N}\p{Nd}]+|[^\s\p{L}\p{N}]+|\s+/gu);

    // Loop through the words to build the Markov chain model
    for (let i = 0; i < words.length - this.order; i++) {
      const state = words.slice(i, i + this.order).join(' ');
      const nextWord = words[i + this.order];
      const sentenceIndex = Math.floor(i / 2); // Adjusted index to consider every other word

      if (!this.chain[state]) {
        this.chain[state] = {};
        this.chain[state].count = 0;
        this.chain[state].nextWords = {};
        this.prior[state] = {}; // Initialize prior distribution for this state
      }

      this.chain[state].count++;
      if (!this.chain[state].nextWords[nextWord]) {
        this.chain[state].nextWords[nextWord] = 0;
        this.prior[state][nextWord] = 1; // Initialize prior probability for this next word
      }
      this.chain[state].nextWords[nextWord]++;
    }

    // Update the prior distribution using the observed data
    for (const state in this.chain) {
      for (const nextWord in this.chain[state].nextWords) {
        this.prior[state][nextWord] = (this.chain[state].nextWords[nextWord] + this.alpha) / (this.chain[state].count + this.beta);
      }
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
      const nextWord = this._sampleFromPrior(this.prior[currentState]);
      result.push(nextWord);
      currentState = result.slice(-this.order).join(' ');
    }

    return result.join(' ');
  }

  _sampleFromPrior(prior) {
    const total = Object.values(prior).reduce((a, b) => a + b, 0);
    const rand = Math.random() * total;
    let sum = 0;
    for (const word in prior) {
      sum += prior[word];
      if (sum >= rand) {
        return word;
      }
    }
  }
}
// Example usage:
const order = 6; // Order of the Markov chain
const chain = new BayesianMarkovChain(order);
let offset = 0;
const batchSize = 100;
const totalExamples = 773000; // Update this with the total number of examples in your dataset
function fetchData(offset) {
    axios
        .get('https://datasets-server.huggingface.co/rows', {
            params: {
                dataset: 'stingning/ultrachat',
                config: 'default',
                split: 'train',
                offset: offset,
                length: batchSize,
            },
        })
        .then((response) => {
            // Handle successful response

          
                response.data.rows.forEach((row) => {
                  row.row.data.forEach(example =>  chain.train(example))
                chain.saveModel('markov_model.json')

                });
                console.log(chain.generate(800,'how can we '))
              
            
            // Check if there are more examples to fetch
            if (offset + batchSize < totalExamples) {
                // Fetch the next batch of examples
                offset += batchSize;
                fetchData(offset);
            } else {
                console.log('Finished fetching all examples.');
            }
           
        })
        
        .catch((error) => {
            // Handle error
            fetchData(offset);
            console.error('Error fetching data:', error);
        });

}



fetchData(batchSize)