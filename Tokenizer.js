
class BPETokenizer {
    constructor() {
        this.vocab = new Set();
        this.tokenToID = new Map();
        this.maxSubwordLength = data.maxTokenLength;
        this.subwordFrequencies = new Map();

    }

    learnVocab(corpus, numIterations) {
        for (const word of corpus) {
            this.updateVocab(word);
        }

        for (let iter = 0; iter < numIterations; iter++) {
            const pairs = this.getSubwordPairs();
            if (pairs.size === 0) break;

            const mostCommonPair = this.findMostCommonPair(pairs);
            if (!mostCommonPair) break;

            this.vocab.add(mostCommonPair);
            corpus = corpus.map(word => word.replace(new RegExp(mostCommonPair, 'g'), ''));
        }

        this.cacheSubwordFrequencies();
        this.buildTokenToID();
    }

    getSubwordPairs() {
        const pairs = new Map();
        for (const word of this.vocab) {
            for (let len = 2; len <= this.maxSubwordLength; len++) {
                for (let i = 0; i < word.length - len + 1; i++) {
                    const subword = word.slice(i, i + len);
                    pairs.set(subword, (pairs.get(subword) || 0) + 1);
                }
            }
        }
        return pairs;
    }

    findMostCommonPair(pairs) {
        let maxCount = -1;
        let mostCommonPair = null;
        for (const [pair, count] of pairs.entries()) {
            if (count > maxCount) {
                mostCommonPair = pair;
                maxCount = count;
            }
        }
        return mostCommonPair;
    }

    updateVocab(word) {
        for (let len = 1; len <= this.maxSubwordLength; len++) {
            for (let i = 0; i < word.length - len + 1; i++) {
               
                const subword = word.slice(i, i + len);
               
                this.vocab.add(subword);
            }
        }
    }

    cacheSubwordFrequencies() {
        for (const word of this.vocab) {
            for (let len = 1; len <= this.maxSubwordLength; len++) {
                for (let i = 0; i < word.length - len + 1; i++) {
                    const subword = word.slice(i, i + len);
                    this.subwordFrequencies.set(subword, (this.subwordFrequencies.get(subword) || 0) + 1);
                }
            }
        }
    }

    buildTokenToID() {
        let nextID = 0;
        for (const token of this.vocab) {
            this.tokenToID.set(token, nextID++);
        }
    }
    purgeVocabulary(targetVocabSize) {
        // Convert the vocabulary Set to an array for sorting
        const sortedVocab = Array.from(this.vocab);
    
        // Sort the vocabulary by frequency and length combined
        sortedVocab.sort((a, b) => {
            // Compare frequencies
            const freqDiff = this.subwordFrequencies.get(b) - this.subwordFrequencies.get(a);
            if (freqDiff !== 0) {
                return freqDiff;
            }
            // If frequencies are equal, prioritize longer tokens
            return b.length - a.length;
        });
    
        // Keep only the top targetVocabSize tokens
        const purgedVocab = sortedVocab.slice(0, targetVocabSize);
    
        // Rebuild the tokenToID map
        const tokenToID = new Map();
        for (let i = 0; i < purgedVocab.length; i++) {
            tokenToID.set(purgedVocab[i], i);
        }
    
        // Update the vocabulary and tokenToID
        this.vocab = new Set(purgedVocab);
        this.tokenToID = tokenToID;
    
        // Log the size of the vocabulary
        console.log("BPE Vocabulary Size:", this.vocab.size);
    
        // Update nextID
        this.nextID = purgedVocab.length;
    }
    
    
    isRelativelyUnique(token, uniqueThreshold) {
        // Calculate the number of other tokens that contain the given token as a substring
        const numContainingTokens = Array.from(this.subwordFrequencies.keys()).filter(t => t.includes(token)).length;
    
        // Calculate the uniqueness ratio
        const uniquenessRatio = numContainingTokens / this.vocab.size;
    
        // Return true if the uniqueness ratio is above the threshold
        return uniquenessRatio < uniqueThreshold;
    }
    
    
    encode(text) {
        const { tokens, tokenIDs } = this.tokenize(text);
        
        data.encoded = tokenIDs
        return tokenIDs;
    }

    decode(tokenIDs) {
        let decodedText = '';
        for (const tokenID of tokenIDs) {
            const token = Array.from(this.tokenToID.keys()).find(key => this.tokenToID.get(key) === tokenID);
            decodedText += token;
        }
        data.decoded = decodedText;
        return decodedText;
    }
    
tokenize(text) {
    const tokens = [];
    const tokenIDs = [];
    
  
    const words = text.match(data.regex)
    
    

    for (const word of words) {
        let tokenizedWord = '';
        let i = 0;
        while (i < word.length) {
            let found = false;
            for (let len = this.maxSubwordLength; len >= 1; len--) {
                if (i + len <= word.length) {
                    const subword = word.slice(i, i + len);
                    if (this.vocab.has(subword)) {
                        tokenizedWord += subword + ' ';
                        tokenIDs.push(this.tokenToID.get(subword));
                        i += len;
                        found = true;
                        break;
                    }
                }
            }
            if (!found) {
                tokenizedWord += word[i];
                tokenIDs.push(this.tokenToID.get(word[i]));
                tokenizedWord += '';
                i++;
            }
        }
        // Trim and split the tokenized word to remove extra spaces
        tokens.push(...tokenizedWord.trim().split(/\s+/));
    }
    return { tokens, tokenIDs };
}


clusterTokens(similarityThreshold) {
    const coOccurrenceMatrix = this.buildCoOccurrenceMatrix(corpus);
    const tokenIDs = Array.from(this.vocab).map(subword => this.tokenToID.get(subword));
    const tfidfMatrix = this.calculateTFIDF(coOccurrenceMatrix, tokenIDs);
    const similarities = this.calculateSimilarities(tfidfMatrix);
    const clusters = this.hierarchicalClustering(similarities, similarityThreshold);
    return clusters;
}

buildCoOccurrenceMatrix(corpus) {
    const coOccurrenceMatrix = new Map();
    for (const word of corpus) {
        const subwords = this.getSubwords(word);
        for (const subword of subwords) {
            if (!coOccurrenceMatrix.has(subword)) {
                coOccurrenceMatrix.set(subword, new Map());
            }
            for (const otherSubword of subwords) {
                if (subword !== otherSubword) {
                    const count = coOccurrenceMatrix.get(subword).get(otherSubword) || 0;
                    coOccurrenceMatrix.get(subword).set(otherSubword, count + 1);
                }
            }
        }
    }
    return coOccurrenceMatrix;
}

calculateTFIDF(coOccurrenceMatrix, tokenIDs) {
    const tfidfMatrix = [];
    for (const tokenID of tokenIDs) {
        const tfidfVector = [];
        const token = Array.from(this.tokenToID.keys()).find(key => this.tokenToID.get(key) === tokenID);
        const coOccurrences = coOccurrenceMatrix.get(token);
        let docFreq = 0;
        for (const subword of coOccurrences.keys()) {
            docFreq += 1;
        }
        for (const otherTokenID of tokenIDs) {
            const otherToken = Array.from(this.tokenToID.keys()).find(key => this.tokenToID.get(key) === otherTokenID);
            const tf = coOccurrences.get(otherToken) || 0;
            const idf = Math.log(corpus.length / docFreq);
            tfidfVector.push(tf * idf);
        }
        tfidfMatrix.push(tfidfVector);
    }
    return tfidfMatrix;
}

calculateSimilarities(tfidfMatrix) {
    const similarities = [];
    for (let i = 0; i < tfidfMatrix.length; i++) {
        const rowSimilarities = [];
        for (let j = 0; j < tfidfMatrix.length; j++) {
            const similarity = this.cosineSimilarity(tfidfMatrix[i], tfidfMatrix[j]);
            rowSimilarities.push(similarity);
        }
        similarities.push(rowSimilarities);
    }
    return similarities;
}

cosineSimilarity(vec1, vec2) {
    let dotProduct = 0, magnitude1 = 0, magnitude2 = 0;
    for (let i = 0; i < vec1.length; i++) {
        dotProduct += vec1[i] * vec2[i];
        magnitude1 += vec1[i] ** 2;
        magnitude2 += vec2[i] ** 2;
    }
    magnitude1 = Math.sqrt(magnitude1);
    magnitude2 = Math.sqrt(magnitude2);
    return dotProduct / (magnitude1 * magnitude2);
}

hierarchicalClustering(similarities, similarityThreshold) {
    // Hierarchical clustering implementation (you can use an existing library or implement it manually)
    // For demonstration purposes, I'll provide a basic example using a threshold-based approach
    const clusters = [];
    for (let i = 0; i < similarities.length; i++) {
        const cluster = [];
        for (let j = i + 1; j < similarities.length; j++) {
            if (similarities[i][j] >= similarityThreshold) {
                cluster.push(j);
            }
        }
        if (cluster.length > 0) {
            cluster.push(i); // Include the current token in the cluster
            clusters.push(cluster);
        }
    }
    return clusters;
}

   
}

// Example usage:
const corpusText = data.corpus;


let text = data.inputText;
const corpus = corpusText.match(data.regex)



const bpeTokenizer = new BPETokenizer();
bpeTokenizer.learnVocab(corpus, data.numMerges);
bpeTokenizer.purgeVocabulary(data.vocabLength); 







function encodeDecode(text){
    
    let tokenized = bpeTokenizer.tokenize(text)
    data.encoded = tokenized.tokenIDs
    data.decoded = tokenized.tokens


}

encodeDecode(data.inputText);


  // Example text data as an array of sentences
var textData = splitTextIntoSentences(corpusText);

// Encode each sentence in the text data
var encodedData = textData.map((sentence) => bpeTokenizer.tokenize(sentence).tokenIDs);
// Function to pad sentences to the same length
function padSentences(sentences, maxLength) {
    return sentences.map((sentence) => {
        const paddingLength = maxLength - sentence.length;
        const paddedSentence = sentence.concat(Array(paddingLength).fill(0));
        return paddedSentence;
    });
}

// Determine the maximum length among all sentences
var maxLength = Math.max(...encodedData.map((sentence) => sentence.length));

// Pad all sentences to the same length
var paddedData = padSentences(encodedData, maxLength);




// Example RBM training on encoded text data
var numVisible = maxLength; // Use the maximum length for numVisible
var numHidden = 80; // Increased hidden units for better representation
var rbm = new RBM(numVisible, numHidden);

let learningRate = 3e-8
// Train RBM on padded encoded data
var rbmData = normalize(paddedData); // RBM expects data in an array


   
      
            
            rbm.train(rbmData, { numEpochs: 10, learningRate: learningRate });
           

        
   
     let activations = rbm.forwardPass(rbmData[3])
     onsole.log('Reconstructed',rbm.reconstructInput(activations.w))

function normalize(arr) {
  const allVals = [].concat.apply([], arr); // Flatten the 2D array
  const min = 0;
  const max = data.vocabLength;

  return arr.map(subArr => subArr.map(x => (x - min) / (max - min)));
}

// Example usage:



function splitTextIntoSentences(text) {
    // Define a regular expression to match sentence endings
    var sentenceEndingsRegex = /[.!?]/;

    // Split the text into sentences based on the regular expression
    var sentences = text.split(sentenceEndingsRegex);

    // Remove empty strings and trim whitespace from each sentence
    sentences = sentences.filter((sentence) => sentence.trim() !== '');

    return sentences;
}





