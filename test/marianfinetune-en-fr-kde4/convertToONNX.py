import torch
from transformers import MarianMTModel

# Load model
model = MarianMTModel.from_pretrained("path_to_checkpoint")
model.eval()

# Dummy inputs
batch_size = 1
src_seq_len = 32
tgt_seq_len = 10

input_ids = torch.randint(0, model.config.vocab_size, (batch_size, src_seq_len))
decoder_input_ids = torch.randint(0, model.config.vocab_size, (batch_size, tgt_seq_len))

# Export
torch.onnx.export(
    model,
    args=({"input_ids": input_ids, "decoder_input_ids": decoder_input_ids},),  # <- FIXED
    f="marianmt.onnx",
    input_names=["input_ids", "decoder_input_ids"],
    output_names=["logits"],
    opset_version=13,
    dynamic_axes={
        "input_ids": {0: "batch", 1: "src_seq_len"},
        "decoder_input_ids": {0: "batch", 1: "tgt_seq_len"},
        "logits": {0: "batch", 1: "tgt_seq_len"},
    }
)
