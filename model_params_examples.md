# Examples for model parameters

## Titan Large

```json
{"maxTokenCount": 512,"stopSequences": [],"temperature":0.1,"topP":0.9}
```

## Jurassic Grande and Jumbo 

```json
{"maxTokens": 200,"temperature": 0.5,"topP": 0.5,"stopSequences": [],"countPenalty": { "scale": 0},"presencePenalty": {"scale": 0},"frequencyPenalty": {"scale": 0}}
```

## Claude


```json
{"max_tokens_to_sample": 300,"temperature": 0.5,"top_k": 250,"top_p": 1,  "stop_sequences": ["\n\nHuman:"]}
```
