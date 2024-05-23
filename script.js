// Sample training data
const trainingData = [
    { input: 'Hello', output: 'Hi!' },
    { input: 'How are you?', output: 'I am good, thank you!' },
    // Add more training data here
  ];
  
  // Convert training data to numerical representation
  function prepareTrainingData(data) {
    const inputs = [];
    const outputs = [];
  
    data.forEach(({ input, output }) => {
      const inputArray = input.split('').map(char => char.charCodeAt(0));
      const outputArray = output.split('').map(char => char.charCodeAt(0));
  
      inputs.push([...inputArray, ...Array(maxInputLength - inputArray.length).fill(0)]);
      outputs.push(outputArray);
    });
  
    return { inputs, outputs };
  }
  
  // Training and saving the model
  async function trainAndSaveModel() {
    const { inputs, outputs } = prepareTrainingData(trainingData);
    await createModel();
    await trainModel(inputs, outputs);
    await saveModel();
  }
  
  // Loading the trained model and making predictions
  async function initializeModelAndPredict() {
    await loadModel();
    const userInput = document.getElementById('user-input');
    const sendButton = document.getElementById('send-button');
    const chatOutput = document.getElementById('chat-output');
  
    sendButton.addEventListener('click', async () => {
      const inputText = userInput.value;
      if (inputText.trim() !== '') {
        const inputData = inputText.split('').map(char => char.charCodeAt(0));
        const predictionData = await predict([...inputData, ...Array(maxInputLength - inputData.length).fill(0)]);
        const predictionText = String.fromCharCode(...predictionData);
        chatOutput.innerHTML += `User: ${inputText}`;
        chatOutput.innerHTML += `Bot: ${predictionText}`;
        userInput.value = '';
        chatOutput.scrollTop = chatOutput.scrollHeight;
      }
    });
  }
  
  // Start the training and prediction process
  trainAndSaveModel().then(initializeModelAndPredict);