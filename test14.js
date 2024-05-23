class MemoryfulMarkovChain {
    constructor(order) {
      this.order = order;
      this.states = {};
    }
  
    addSequence(sequence) {
      for (let i = this.order; i < sequence.length; i++) {
        const prevState = sequence.slice(i - this.order, i);
        const nextState = sequence[i];
  
        const prevStateKey = prevState.join(',');
        if (!this.states[prevStateKey]) {
          this.states[prevStateKey] = {};
        }
  
        if (!this.states[prevStateKey][nextState]) {
          this.states[prevStateKey][nextState] = 0;
        }
  
        this.states[prevStateKey][nextState]++;
      }
    }
  
    getNextState(prevState) {
      const prevStateKey = prevState.join(',');
      const possibleNextStates = this.states[prevStateKey];
  
      if (!possibleNextStates) {
        return null;
      }
  
      const nextStateOptions = Object.keys(possibleNextStates);
      const nextStateWeights = nextStateOptions.map(state => possibleNextStates[state]);
  
      const totalWeight = nextStateWeights.reduce((acc, weight) => acc + weight, 0);
      let randomWeight = Math.random() * totalWeight;
  
      for (let i = 0; i < nextStateOptions.length; i++) {
        if (randomWeight < nextStateWeights[i]) {
          return nextStateOptions[i];
        }
        randomWeight -= nextStateWeights[i];
      }
  
      return null; // Should not reach here
    }
  
    generateSequence(startState, length) {
      let prevState = startState.slice(startState.length - this.order);
  
      const sequence = [...startState];
  
      for (let i = 0; i < length; i++) {
        const nextState = this.getNextState(prevState);
        if (!nextState) {
          break; // No valid next state found
        }
        sequence.push(nextState);
        prevState.shift();
        prevState.push(nextState);
      }
  
      return sequence;
    }
  }
  
  // Example usage:
  const order = 7;
  const markovChain = new MemoryfulMarkovChain(order);
  
  // Train the model with some example sequences
  const sequences = [
    ['A', 'B', 'C', 'A', 'B', 'C'],
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['X', 'Y', 'Z', 'X', 'Y', 'Z'],
  ];
  
  sequences.forEach(sequence => markovChain.addSequence(sequence));
  
  // Generate a sequence based on a starting state
  const startingState = ['Y', 'Z','X'];
  const generatedSequence = markovChain.generateSequence(startingState, 10);
  console.log('Generated Sequence:', generatedSequence.join(','));
  