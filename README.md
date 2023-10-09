# Ask Amazon Bedrock

Converse with your favorite [Amazon Bedrock](https://aws.amazon.com/bedrock/) large language model from the command line.

<p>
  <img width="1000" src="README.svg">
</p>

Its main added value is that it locally persists AWS account and model configuration to enable quick and easy access.

## Installation

⚠️ Requires Python >= 3.9

⚠️ Requires a working AWS CLI setup configured with a profile that allows Amazon Bedrock access. See [CLI documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html) for details.

```
pip install ask-bedrock
```

## Usage

To start a conversation, simply enter the following command:

```
ask-bedrock converse
```

Upon first run, you will be led through a configuration flow.

If you’re fully configured, the tool will show you a `>>>` prompt and you can start interacting with the configured model.

Multi-line prompts can be wrapped into `<<< >>>` blocks.

To end your interaction, hit `Ctrl + D`. Note that the conversation will be lost.

## Configuration

Ask Amazon Bedrock stores your user configuration in `$HOME/.config/ask-bedrock/config.yaml`. This file may contain several sets of configuration (contexts). For instance, you can use contexts to switch between different models. Use the `--context` parameter to select the context you'd like to use. The default context is `default`.

If no configuration is found for a selected context, a new one is created. If you want to re-configure an existing config, use 

```
ask-bedrock configure --context mycontext
```

### Model parameters

This JSON is passed to Langchain during client setup (as `model_kwargs`). The schema depends on the model that is used. Have a look at the [examples](model_params_examples.md).

If you want to configure multiple lines, model parameters can be wrapped in `<<< >>>`.

## Build from source

```
pip install -r requirements.txt
python -m build
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

