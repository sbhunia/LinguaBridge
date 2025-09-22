import evaluate

metric = evaluate.load("sacrebleu")

predictions = [
    "This plugin lets you translate web pages between several languages automatically."
]
references = [
    [
        "This plugin allows you to automatically translate web pages between several languages."
    ]
]
print(metric.compute(predictions=predictions, references=references))
