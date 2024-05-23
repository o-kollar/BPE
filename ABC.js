const axios  = require("axios");


function matrix(){

    this.init = function(y, x){
        var M = new Array(y);
        for (var k = 0; k < y ; k ++)
        	M[k] = new Array(x);
        return M ;
     }
}

matrix.prototype.zeros = function(y, x){
    	var M = this.init(y, x);
  	    for(var i = 0; i < y; i++){
	        for(var j = 0; j < x; j++)
	            M[i][j] = 0; 
        } 
        return M ;  
    } 

matrix.prototype.random = function(y, x){
    	var M = this.init(y, x);
  	    for(var i = 0; i < y; i++){
	        for(var j = 0; j < x; j++)
	            M[i][j] = Math.random(); 
        } 
        return M ;  
    }  

matrix.prototype.dot = function(A, B){
	if (B[0].length === undefined)
		var l = 1;
	else
		var l = B[0].length;
    if (A[0].length === undefined)
		var r = 1;
	else
        var r = A[0].length;
	var C = this.zeros(A.length, l);
	for(var i = 0; i < A.length;i ++){
		for(var j = 0; j < l;j ++){
			for(var s = 0; s < r;s ++){
				
                C[i][j] += A[i][s] * B[s][j];

            }
        }
    }
    return C;
}

matrix.prototype.add = function(A, B){

	if (B[0].length === undefined)
		var l = 1;
	else
		var l = B[0].length;
    if (A[0].length === undefined)
		var r = 1;
	else
        var r = A[0].length;

	if(A.length <= B.length)
		var s = B.length;
	else
		var s = A.length;
	if(r <= l)
		var d = l;
	else 
		var d = r;
    var C = this.zeros(s, d);
	for (var i = 0;i < s;i++){
		for(var j = 0;j < d;j++){
			if(A[i] !== undefined){
				if(A[i][j]!== undefined)
				C[i][j] += A[i][j];
		     }
			if(B[i]!== undefined){
				if(B[i][j]!== undefined)
				C[i][j] += B[i][j];
		     }
		} 
	}
    return C;   
}

matrix.prototype.sigmoid = function(A){

	if (A[0].length === undefined)
		var l = 1;
	else
		var l = A[0].length;
    
    var C = this.zeros(A.length, l);
	for (var i = 0;i < A.length;i++){
		for(var j = 0;j < l;j++){
			C[i][j] = 1 / (1 + Math.exp(-1 * A[i][j]));
		} 
	}
    return C;   
}

matrix.prototype.transpose = function(A){
	if (A[0].length === undefined)
		var l = 1;
	else
		var l = A[0].length;
    
    var C = this.zeros(l, A.length);
	for (var i = 0;i < A.length;i++){
		for(var j = 0;j < l;j++){
			C[j][i] = A[i][j];          																																																																																																																																																																																																																																																																																				
		} 
	}
    return C;   
}
var matrix = new matrix();

function rnn(input){
    var input = input;
    var rate = 0.05;
    var w_x = Math.random() - 0.5 ;
    var w_y = Math.random() - 0.5 ;
    var b = Math.random() - 0.5 ;
    var l = input[0].length;
    var hidden_input = matrix.zeros(1, l);

    var output = matrix.zeros(1, l);
    var delta ;
    
    for (var i = 0; i < 100000; i++) {

      for (var j = 0; j < l -1 ; j++){
         hidden_input[0][j] = w_x * input[0][j] + w_y * output[0][j] + b;
         output[0][j + 1] = 1 / (1 + Math.exp(-1* hidden_input[0][j])); 
      }
     
      for (var j = 0; j < l - 1 ; j++){

          for (var t = 0; t < j + 1; t++){
             var d = j - t;
             if (t == 0){
               delta = output[0][d + 1] - input[0][d + 1];
                  
             }

             else{
               delta = w_y * output[0][d + 1] * ( 1 - output[0][d + 1]) * delta ;
               }

               w_x = w_x - rate * delta * input[0][d];
               w_y = w_y - rate * delta * output[0][d];
               b = b - rate * delta ;
             
         }
      }
   }
   for (var j = 0; j < l - 1 ; j++){
         hidden_input[0][j] = w_x * input[0][j] + w_y * output[0][j] + b;
         output[0][j + 1] = 1 / (1 + Math.exp(-1 * hidden_input[0][j])); 
      }

 return output;
}


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
                console.log(mostFrequentPair);
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


vocab = new Vocab()
let offset = 0;
const batchSize = 100;
const totalExamples = 52000; // Update this with the total number of examples in your dataset
function fetchData(offset) {
    axios
        .get('https://datasets-server.huggingface.co/rows', {
            params: {
                dataset: 'NickyNicky/oasst2_orpo_mix_54k_chatML_phi_3',
                config: 'default',
                split: 'train',
                offset: offset,
                length: batchSize,
            },
        })
        .then((response) => {
            // Handle successful response

            for (let i = 0; i < 1; i++) {
                
                response.data.rows.forEach((row) => {
                    
                    let lr = 0.1
                       let tokens = vocab.tokenize(row.row.chosen)
                       
                       for(let i =0;i<tokens.length;i++){
                        F = rnn([tokens])
                       console.log("Output",F)

                       }
                      
                 
                   
                    
                       

                
                    
                });
            }
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

                   
       




