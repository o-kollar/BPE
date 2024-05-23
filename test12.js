class NonLinearStateSpaceModel {
    constructor(initialState, processNoise, measurementNoise) {
        this.state = initialState;
        this.processNoise = processNoise;
        this.measurementNoise = measurementNoise;
    }

    // Define the process function to describe how the state evolves over time
    processFunction(state) {
        // Example: Non-linear dynamics
        const nextState = {
            x: state.x + Math.sin(state.y),
            y: state.y + Math.cos(state.x),
        };
        return nextState;
    }

    // Define the measurement function to describe how measurements are obtained from the state
    measurementFunction(state) {
        // Example: Non-linear measurement
        const measurement = {
            z: Math.sqrt(state.x ** 2 + state.y ** 2) + this.measurementNoise * Math.random(),
        };
        return measurement;
    }

    // Predict the next state based on the current state and process noise
    predict() {
        // Apply process function to predict next state
        this.state = this.processFunction(this.state);

        // Add process noise to the predicted state
        this.state.x += this.processNoise * Math.random();
        this.state.y += this.processNoise * Math.random();
    }

    // Update the state estimate based on a new measurement
    update(measurement) {
        // Apply measurement function to estimate state from measurement
        const estimatedState = {
            x: measurement.z * Math.cos(this.state.y),
            y: measurement.z * Math.sin(this.state.x),
        };

        // Update state estimate with measurement
        this.state = estimatedState;
    }
}

// Example usage:
const initialState = { x: 0, y: 0 };
const processNoise = 0.1;
const measurementNoise = 0.1;

const model = new NonLinearStateSpaceModel(initialState, processNoise, measurementNoise);

for (let i = 0; i < 10; i++){
    // Predict next state
model.predict();


// Generate a measurement
const measurement = model.measurementFunction(model.state);

// Update state estimate based on measurement

model.update({ z: 3.8199605225825097 });
console.log('Measurement:',{ z: 3.8199605225825097 });

console.log('Predicted state:', model.state);
}
