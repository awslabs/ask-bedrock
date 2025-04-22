# Ask Amazon Bedrock

Converse with your favorite [Amazon Bedrock](https://aws.amazon.com/bedrock/) large language model from the command line.

<p>
  <img width="1000" src="https://raw.githubusercontent.com/awslabs/ask-bedrock/main/README.svg">
</p>

This tool is a wrapper around the low-level Amazon Bedrock APIs. Its main added value is that it locally persists AWS account and model configuration to enable quick and easy interaction.

## Installation

⚠️ Requires Python >= 3.9

⚠️ Requires a working AWS CLI setup configured with a profile that allows Amazon Bedrock access. See [CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) for details.


```shell
pip install ask-bedrock
```

You can also build/run this project locally, see [Building and Running Locally](#building-and-running-locally).

## Usage

### Activating models

Before you can use this command line tool, you need to [request model access through the AWS Console](https://console.aws.amazon.com/bedrock/home#/modelaccess) in a region where Bedrock is available: Switch to the region where you want to run Bedrock, go to ”Model access“, click “Edit”, activate the models you wish to use, and then click “Save changes”.

### Invocation

To start a conversation, simply enter the following command:

```shell
ask-bedrock converse
```

If you don't need a conversation, you can get a simple request-response using:

```shell
ask-bedrock prompt "What's up?"
```

Upon the first run, you will be led through a configuration flow. To learn more about configuration options, see the [Configuration](#configuration) section below.

If you’re fully configured, the tool will show you a `>>>` prompt and you can start interacting with the configured model.

Multi-line prompts can be wrapped into `<<< >>>` blocks.

To end your interaction, hit `Ctrl + D`. Note that the conversation will be lost.

You can also use a single prompt with a simple request-response:
```
ask-bedrock prompt "complete this sentence: One small step for me"
```

### Pricing

Note that using Ask Amazon Bedrock incurs AWS fees. For more information, see [Amazon Bedrock pricing](https://aws.amazon.com/bedrock/pricing/). Consider using a dedicated AWS account and [AWS Budgets](https://docs.aws.amazon.com/cost-management/latest/userguide/budgets-managing-costs.html) to control costs.

## Configuration

*Ask Amazon Bedrock* stores your user configuration in `$HOME/.config/ask-bedrock/config.yaml`. This file may contain several sets of configuration (contexts). For instance, you can use contexts to switch between different models. Use the `--context` parameter to select the context you'd like to use. The default context is `default`.

If no configuration is found for a selected context, a new one is created. If you want to change an existing config, use

```shell
ask-bedrock configure --context mycontext
```

You can also create or edit the configuration file yourself in `$HOME/.config/ask-bedrock/config.yaml`:

```yaml
contexts:
  default:
    region: ""                  # an AWS region where you have activated Bedrock
    aws_profile: ""             # a profile from your ~/.aws/config file
    model_id: ""                # a Bedrock model, e.g. "ai21.j2-ultra-v1"
    inference_config: "{}"      # a JSON object with inference configuration
```

### Inference Configuration

The `inference_config` is passed directly to the Amazon Bedrock Runtime `converse_stream` API. This configuration controls the behavior of model generation, including parameters like temperature and token limits.

Common parameters include:

- `temperature` (float): Controls randomness in response generation. Lower values make responses more deterministic.
- `topP` (float): Controls diversity of responses by considering tokens with top cumulative probability.
- `maxTokens` (integer): Maximum number of tokens to generate in the response.
- `stopSequences` (array): Sequences where the model should stop generating.

Example configurations:

```json
{
  "temperature": 0.7,
  "topP": 0.9,
  "maxTokens": 3000
}
```

```json
{
  "temperature": 0.5,
  "maxTokens": 500,
  "stopSequences": ["\n\n"]
}
```

For more details, see the [Amazon Bedrock Runtime InferenceConfiguration API Reference](https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_InferenceConfiguration.html).

## Building and Running Locally

```
pip install build
python -m build
pip install -e .
ask_bedrock converse
```


## Troubleshooting

**Q:** The model responses are cut off mid-sentence.

**A:** Configure the model to allow for longer response by increasing the `maxTokens` value in the inference configuration (see above). For example: `{"maxTokens": 3000}`

---

**Q**: I'm getting an error that is not listed here.

**A**: Use the `--debug` option to find out more about the error. If you cannot solve it, create an issue.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
