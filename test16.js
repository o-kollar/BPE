const axios = require('axios')


              
               const postData = {
                "prompt": `<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a youthful helpful assistant named K8<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\nWhat is the tallest building in the world ?<|eot_id|>\n`,
                "model": "meta/meta-llama-3-70b-instruct",
                "systemPrompt": "You are a helpful assistant.",
                "temperature": 0.75,
                "topP": 0.9,
                "maxTokens": 200,
                "image": null,
                "audio": null
               }

               axios.post(' https://www.llama2.ai/api', postData)
  .then(response => {
    // Handle successful response

   
    

console.log(response)
  })