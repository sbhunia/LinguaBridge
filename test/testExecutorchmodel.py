# https://github.com/huggingface/transformers/pull/36486
# model = ExecuTorchModelForSeq2SeqLM.from_pretrained("google-t5/t5-small", recipe="xnnpack")
# generated_text = model.text_generation(
#     tokenizer=AutoTokenizer.from_pretrained("google-t5/t5-small"),
#     prompt="summarize: Simply put, the theory of relativity states that ...",
# )
# https://github.com/huggingface/optimum-executorch?tab=readme-ov-file#option-2-export-and-load-separately

# --------------------------------------------------------------------------------
# from optimum.executorch import ExecuTorchModelForSeq2SeqLM
# from transformers import AutoTokenizer

# # Load the exported model
# model = ExecuTorchModelForSeq2SeqLM.from_pretrained("/home/yogesh/LinguaBridge/test/optimum-executorch/executorch-export")

# # Initialize tokenizer and generate text
# tokenizer = AutoTokenizer.from_pretrained("/home/yogesh/LinguaBridge/test/t5-small-60m/t5-fine-tuned/checkpoint-8340")
# generated_text = model.text_generation(
#     tokenizer=tokenizer,
#     prompt="translate english to french: 'Hello, my name is Jack.'"
# )
# print(generated_text)
# --------------------------------------------------------------------------------

# test
# --------------------------------------------------------------------------------

import torch
# from executorch.runtime import Runtime

# runtime = Runtime.get()

# input_tensor = torch.randn(1, 3, 32, 32)
# program = runtime.load_program("/home/yogesh/LinguaBridge/test/optimum-executorch/executorch-export/model.pte")
# method = program.load_method("forward")
# outputs = method.execute([input_tensor, input_tensor])

model = torch.jit.load("/home/yogesh/LinguaBridge/test/optimum-executorch/executorch-export/model.pte")
x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print(model(x))