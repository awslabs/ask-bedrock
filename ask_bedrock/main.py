import atexit
import json
import logging
import os
import sys
from collections.abc import Callable

import boto3
import click
import yaml

logger = logging.getLogger(__name__)
if not logger.handlers:
    logger.addHandler(handler := logging.StreamHandler())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.propagate = False

logging.basicConfig(level=logging.INFO)

config_file_path = os.path.join(
    os.path.expanduser("~"), ".config", "ask-bedrock", "config.yaml"
)

atexit.register(
    lambda: logger.debug(
        "\nThank you for using Ask Amazon Bedrock! Consider sharing your feedback here: https://pulse.aws/survey/GTRWNHT1"
    )
)


@click.group()
def cli():
    pass


def log_error(msg: str, e: Exception = None):
    logger.error(click.style(msg, fg="red"))
    if e:
        logger.debug(e, exc_info=True)
        logger.error(click.style(str(e), fg="red"))


@cli.command()
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
def converse(context: str, debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = init_config(context)

    start_conversation(config)


@cli.command()
@click.argument("input")
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
def prompt(input: str, context: str, debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = init_config(context)

    try:
        bedrock_runtime = get_bedrock_runtime(config)
        model_id = config["model_id"]
        inference_config = (
            json.loads(config["inference_config"])
            if "inference_config" in config
            else {}
        )
    except Exception as e:
        log_error("Error while initializing Bedrock client", e)
        return

    try:
        # Format according to the Converse API specification
        messages = [{"role": "user", "content": [{"text": input}]}]
        stream_response(bedrock_runtime, model_id, messages, inference_config)
    except Exception as e:
        log_error("Error while generating response", e)


@cli.command()
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
def configure(context: str, debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    existing_config = get_config(context)
    config = create_config(existing_config)
    if config is not None:
        put_config(context, config)


def start_conversation(config: dict):
    try:
        bedrock_runtime = get_bedrock_runtime(config)
        model_id = config["model_id"]
        inference_config = (
            json.loads(config["inference_config"])
            if "inference_config" in config
            else {}
        )
    except Exception as e:
        log_error("Error while initializing Bedrock client", e)
        return

    # Initialize conversation memory
    conversation_memory = []

    while True:
        prompt = multiline_prompt(
            lambda: click.prompt(click.style(">>>", fg="green")), return_newlines=True
        )

        # Add user message to conversation memory
        conversation_memory.append({"role": "user", "content": [{"text": prompt}]})

        try:
            stream_response(
                bedrock_runtime, model_id, conversation_memory, inference_config
            )
            # Response is captured and added inside stream_response function
        except Exception as e:
            log_error("Error while generating response", e)
            continue


def get_config(context: str) -> dict:
    if not os.path.exists(config_file_path):
        return None
    with open(config_file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not "contexts" in config:
        return None
    if not context in config["contexts"]:
        return None
    return config["contexts"][context]


# Stores a config for a given context physically
def put_config(context: str, new_config: dict):
    if os.path.exists(config_file_path):
        with open(config_file_path, "r", encoding="utf-8") as f:
            current_config_file = yaml.safe_load(f)
    else:
        os.makedirs(os.path.dirname(config_file_path), exist_ok=True)
        current_config_file = {"contexts": {}}
    new_contexts = current_config_file["contexts"] | {context: new_config}
    new_config_file = current_config_file | {"contexts": new_contexts}

    with open(config_file_path, "w", encoding="utf-8") as f:
        logger.info(f"Writing configuration to {config_file_path}.")
        f.write(yaml.dump(new_config_file))


# Leads through a new configuration dialog
def create_config(existing_config: str) -> dict:
    available_profiles = click.Choice(boto3.session.Session().available_profiles)
    if len(available_profiles.choices) == 0:
        log_error(
            "No profiles found. Make sure you have configured the AWS CLI with at least one profile."
        )
        return None
    aws_profile = click.prompt(
        "👤 AWS profile",
        type=available_profiles,
        default=existing_config["aws_profile"] if existing_config else None,
    )
    region = click.prompt(
        "🌍 Bedrock region",
        default=existing_config["region"] if existing_config else None,
    )

    bedrock = boto3.Session(profile_name=aws_profile).client("bedrock", region)

    try:
        all_models = bedrock.list_foundation_models()["modelSummaries"]
    except Exception as e:
        log_error("Error listing foundation models", e)
        return None

    applicable_models = [
        model
        for model in all_models
        if model["outputModalities"] == ["TEXT"]
        and "TEXT" in model["inputModalities"]  # multi-modal input models are allowed
        and "ON_DEMAND" in model["inferenceTypesSupported"]
        and model["responseStreamingSupported"]  # Only include streaming-capable models
    ]

    available_models = click.Choice([model["modelId"] for model in applicable_models])
    model_id = click.prompt(
        "🚗 Model",
        type=available_models,
        default=existing_config["model_id"] if existing_config else None,
    )

    # Handle migration from old format if needed
    default_inference_config = "{}"
    if existing_config:
        if "inference_config" in existing_config:
            default_inference_config = existing_config["inference_config"]
        elif "model_params" in existing_config:
            # Try to migrate parameters from old format
            old_params = json.loads(existing_config["model_params"])
            migrated_config = migrate_model_params(model_id, old_params)
            default_inference_config = json.dumps(migrated_config)

    inference_config = multiline_prompt(
        lambda: click.prompt(
            "🔠 Inference configuration (JSON)",
            default=default_inference_config,
        ),
        return_newlines=False,
    )

    config = {
        "region": region,
        "aws_profile": aws_profile,
        "model_id": model_id,
        "inference_config": inference_config,
    }

    # Test the model
    bedrock_runtime = get_bedrock_runtime(config)
    test_message = [
        {
            "role": "user",
            "content": [
                {
                    "text": "You are an assistant used in a CLI tool called 'Ask Bedrock'. "
                    "The user has just completed their configuration. Write them a nice hello message, "
                    "including saying that it is from you."
                }
            ],
        }
    ]

    try:
        stream_response(
            bedrock_runtime, model_id, test_message, json.loads(inference_config)
        )
    except Exception as e:
        if "AccessDeniedException" in str(e):
            click.secho(
                f"{str(e)}\nAccess denied while trying out the model. Have you enabled model access? "
                f"Go to the Amazon Bedrock console and select 'Model access' to make sure. "
                f"Alternatively, choose a different model.",
                fg="red",
            )
            return None
        else:
            click.secho(
                f"{str(e)}\nSomething went wrong while trying out the model, not saving this.",
                fg="red",
            )
            return None

    return config


def migrate_model_params(model_id, old_params):
    """Migrate model parameters from old format to new Converse API format."""
    logger.info(
        f"Migrating model parameters for model {model_id} to Converse API format"
    )

    # Default empty config - only use parameters supported by Converse API
    inference_config = {}

    # Common parameter mappings for all models
    if "temperature" in old_params:
        inference_config["temperature"] = old_params["temperature"]

    if "top_p" in old_params:
        inference_config["topP"] = old_params["top_p"]
    elif "topP" in old_params:
        inference_config["topP"] = old_params["topP"]

    # Map token limits - maxTokens is the standard in Converse API
    if "max_tokens_to_sample" in old_params:
        inference_config["maxTokens"] = old_params["max_tokens_to_sample"]
    elif "max_tokens" in old_params:
        inference_config["maxTokens"] = old_params["max_tokens"]
    elif "maxTokenCount" in old_params:
        inference_config["maxTokens"] = old_params["maxTokenCount"]
    elif "maxTokens" in old_params:
        inference_config["maxTokens"] = old_params["maxTokens"]

    # Map stop sequences
    if "stop_sequences" in old_params and old_params["stop_sequences"]:
        inference_config["stopSequences"] = old_params["stop_sequences"]
    elif "stopSequences" in old_params and old_params["stopSequences"]:
        inference_config["stopSequences"] = old_params["stopSequences"]

    # Remove model-specific parameters that aren't supported in Converse API
    # We need to drop parameters like anthropic_version

    logger.info(f"Migrated config: {inference_config}")
    return inference_config


# Tries to find a config, creates one otherwise
def init_config(context: str) -> dict:
    config = get_config(context)
    if not config:
        click.echo(
            f"No configuration found for context {context}. Creating new configuration."
        )
        config = create_config(None)
        put_config(context, config)

    # Handle migration from older config formats
    if config and "model_params" in config and "inference_config" not in config:
        # Migrate model parameters to inference_config format
        model_id = config["model_id"]
        old_params = json.loads(config["model_params"])
        inference_config = migrate_model_params(model_id, old_params)

        config["inference_config"] = json.dumps(inference_config)
        logger.info(
            f"Migrated model_params to inference_config format for context {context}"
        )
        put_config(context, config)

    return config


def get_bedrock_runtime(config: dict):
    credentials_profile_name = config["aws_profile"]
    region = config["region"]
    return boto3.Session(profile_name=credentials_profile_name).client(
        "bedrock-runtime", region
    )


def get_bedrock(config: dict):
    credentials_profile_name = config["aws_profile"]
    region = config["region"]
    return boto3.Session(profile_name=credentials_profile_name).client(
        "bedrock", region
    )


def stream_response(bedrock_runtime, model_id, messages, inference_config):
    try:
        response = bedrock_runtime.converse_stream(
            modelId=model_id,
            messages=messages,
            inferenceConfig=inference_config if inference_config else None,
        )

        full_response = ""
        # The response is a stream of events
        for event in response["stream"]:
            # Process content delta events which contain the actual text
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    chunk_text = delta["text"]
                    full_response += chunk_text
                    sys.stdout.write(click.style(chunk_text, fg="yellow"))
                    sys.stdout.flush()

        # Add the assistant's response to conversation memory
        if messages and isinstance(messages, list):
            messages.append({"role": "assistant", "content": [{"text": full_response}]})

        sys.stdout.write("\n")
        sys.stdout.flush()

        return full_response
    except Exception as e:
        log_error(f"Error in stream_response: {str(e)}", e)
        raise


def multiline_prompt(prompt: Callable[[], str], return_newlines: bool) -> str:
    response = prompt()
    if response.startswith("<<<"):
        response = response[3:]
        newlines = "\n" if return_newlines else ""
        while not response.endswith(">>>"):
            response += newlines + prompt()
        response = response[:-3]
    return response


if __name__ == "__main__":
    cli()
