# Ask Amazon Bedrock

Converse with your favorite [Amazon Bedrock](https://aws.amazon.com/bedrock/) large language model from the command line.

<p>
  <img width="1000" src="https://raw.githubusercontent.com/awslabs/ask-bedrock/main/README.svg">
</p>

This tool is a wrapper around the low-level Amazon Bedrock APIs and [Langchain](https://python.langchain.com/docs/integrations/llms/bedrock). Its main added value is that it locally persists AWS account and model configuration to enable quick and easy interaction.

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

Upon the first run, you will be led through a configuration flow. To learn more about configuration options, see the [Configuration](#configuration) section below.

If you’re fully configured, the tool will show you a `>>>` prompt and you can start interacting with the configured model.

Multi-line prompts can be wrapped into `<<< >>>` blocks.

To end your interaction, hit `Ctrl + D`. Note that the conversation will be lost.

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
    model_params: "{}"          # a JSON object with parameters for the selected model
```

### Model parameters

This JSON is passed to Langchain during client setup (as `model_kwargs`). The schema depends on the model that is used. Have a look at the [examples](model_params_examples.md).

If you want to configure multiple lines, model parameters can be wrapped in `<<< >>>`.

## Building and Running Locally

```
pip install build
python -m build
python ask_bedrock/main.py converse
```

## Feedback

As this tool is still early stage, we are very interested in hearing about your experience. Please take one minute to take a little survey: **https://pulse.aws/survey/GTRWNHT1**


## Troubleshooting

**Q:** I’m getting the following error during invocation: “ValueError: Error raised by bedrock service: An error occurred (AccessDeniedException) when calling the InvokeModel operation: Your account is not authorized to invoke this API operation.”

**A:** You may have selected a model that is currently not yet activated for public usage. It may have been listed it in the selection of available models, but unfortunately some models (such as Amazon Titan) aren’t yet available via API.

---

**Q:** The model responses are cut off mid-sentence.

**A:** Configure the model to allow for longer response. Use model parameters (see above) for this. Claude for example would take the following model parameters: `{"max_tokens_to_sample": 3000}`

---

**Q**: I'm getting an error that is not listed here.

**A**: Use the `--debug` option to find out more about the error. If you cannot solve it, create an issue.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

