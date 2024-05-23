
var nj = require('numjs')

class OnehotEncoder {
    constructor(vocabSize, mapping) {
        this.vocabSize = vocabSize
        this.mapping = mapping
    }

    encode(char) {
        var charEncoding = new Array(this.vocabSize).fill(0)
        var idx = this.mapping[char]
        charEncoding[idx] = 1
        return charEncoding
    }
}

function _randomChoice(arr, p) {
    var rnd = p.reduce( (a, b) => a + b ) * Math.random();
    var idx = p.findIndex( a => (rnd -= a) < 0 );
    return arr[idx]
}

// replication of numpy.random.choice
function randomChoice(arr, p, count=1) {
    return Array.from(Array(count), _randomChoice.bind(null, arr, p));
}

// create 2d tensor with flattened input as diagonal
function diagFlat(x) {
    var x = x.flatten()
    var id = nj.identity(x.shape[0])
    var xTensor = nj.stack(Array(x.shape[0]).fill(x))
    return nj.multiply(xTensor, id)
}

function _iterator(x, fn, ...args) {
    var out = x.flatten().tolist()

    for (let i = 0; i < out.length; i++) {
        // + 0 removes negative sign from 0
        out[i]= fn(out[i], ...args) + 0
    }
    return nj.array(out).reshape(x.shape)
}

module.exports = {
    OnehotEncoder,
    randomChoice,
    diagFlat,
    _iterator
    }