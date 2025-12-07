Summary of Changes
    Removed Prefix: No need to teach the model a new tag like >>mia<<. It will naturally learn that the output domain is Myaamia.

    Removed BERTScore: It is computationally heavy and, more importantly, lang='fr' creates misleading scores for Myaamia text. Added ChrF (Character F-score) instead, which is very robust for languages with complex morphology or spelling variations.

    Dynamic Padding: Removed padding='max_length' in preprocessing. The DataCollator now pads every batch to the length of the longest sentence in that specific batch. This makes training 20-40% faster.

    Label Handling in Metrics: Added np.where in compute_metrics to handle -100 indices (which PyTorch uses to ignore loss calculation on padding). Without this, the decoder might crash or produce garbage when trying to decode the labels for metric calculation.

    Special Tokens Note: Added a commented-out section where you can add special Myaamia characters (like š) if your dataset uses them and they aren't in the standard French alphabet.