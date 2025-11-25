import React, { useEffect, useState } from "react";
import { View, Text, TextInput, Button, ActivityIndicator } from "react-native";
// Import the hooks you actually need
import { File, Paths } from "expo-file-system";
import { initLlama } from 'llama.rn'

export default function Index() {
  const [inputText, setInputText] = useState("How are you?");
  const [outputText, setOutputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [isModelReady, setIsModelReady] = useState(false);

  // --- 1. Set up your local model path ---
  // This is correct!
  const modelFile = new File(Paths.document, 'llama3_1b_finetune_Q4_K_M.gguf');
  const modelURI = modelFile.uri;
  console.log("Loading model from URI:", modelURI);


  
  const handleTranslate = async () => {
    const context = await initLlama({
      model: modelURI,
      use_mlock: true,
      n_ctx: 2048,
      n_gpu_layers: 99, // number of layers to store in GPU memory (Metal/OpenCL)
      // embedding: true, // use embedding
    })

    const stopWords = ['</s>', '<|end|>', '<|eot_id|>', '<|end_of_text|>', '<|im_end|>', '<|EOT|>', '<|END_OF_TURN_TOKEN|>', '<|end_of_turn|>', '<|endoftext|>']

// Do chat completion
const msgResult = await context.completion(
  {
    messages: [
      {
        role: 'system',
        content: 'You are an expert translator. Translate the given English text into Myaamia.',
      },
      {
        role: 'user',
        content: `translate english to myaamia: ${inputText}`,
      },
    ],
    n_predict: 100,
    stop: stopWords,
    // ...other params
  },
  (data) => {
    // This is a partial completion callback
    const { token } = data
  },
)
setOutputText(msgResult.text)
  }

  return (
    <View style={{ flex: 1, justifyContent: "center", padding: 20 }}>
      <Text style={{ fontSize: 22, fontWeight: "bold", marginBottom: 10 }}>
        Llama Translator (EN → MIA)
      </Text>

      <TextInput
        style={{
          borderWidth: 1,
          borderColor: "#888",
          padding: 10,
          marginBottom: 15,
          borderRadius: 6,
        }}
        value={inputText}
        onChangeText={setInputText}
        placeholder="Enter English text"
      />

      <Button
        title="Translate"
        onPress={handleTranslate} // Use your handleTranslate function
        disabled={!isModelReady || loading} // Disable button if not ready
      />

      {loading && <ActivityIndicator size="large" style={{ marginTop: 20 }} />}

      {outputText !== "" && (
        <Text style={{ marginTop: 20, fontSize: 18 }}>
          ✅ Myaamia: {outputText}
        </Text>
      )}
    </View>
  );
}