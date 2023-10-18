import atexit
import json
import os
import sys
from collections.abc import Callable

import boto3
import click
import yaml
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import ConversationChain
from langchain.llms import Bedrock
from langchain.memory import ConversationBufferMemory

config_file_path = os.path.join(
    os.path.expanduser("~"), ".config", "ask-bedrock", "config.yaml"
)

atexit.register(
    lambda: click.echo(
        "\nThank you for using Ask Amazon Bedrock! Consider sharing your feedback here: https://pulse.aws/survey/GTRWNHT1"
    )
)


@click.group()
def cli():
    pass


@cli.command()
@click.option("--context", default="default")
def converse(context: str):
    config = get_config(context)
    if not config:
        click.echo(
            f"No configuration found for context {context}. Creating new configuration."
        )
        config = create_config(None)
        put_config(context, config)

    sts = boto3.client('sts')
    try:
        sts.get_caller_identity()
    except Exception as e:
        click.secho(f"Credentials are NOT valid: {e}. You may need to check your local AWS CLI configuration.")
        return

    start_conversation(config)


@cli.command()
@click.option("--context", default="default")
def configure(context: str):
    existing_config = get_config(context)
    config = create_config(existing_config)
    if config is not None:
        put_config(context, config)


def start_conversation(config: dict):
    try:
        llm = model_from_config(config)
    except Exception as e:
        click.secho(f"Error while building Bedrock model:\n{e}", fg="red")
        return

    conversation = ConversationChain(
        llm=llm,
        memory=ConversationBufferMemory(ai_prefix="Assistant"),
    )

    while True:
        prompt = multiline_prompt(
            lambda: click.prompt(click.style(">>>", fg="green")), return_newlines=True
        )

        response = conversation.predict(input=prompt)

        if not llm.streaming:
            click.secho(response, fg="yellow")


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
        click.echo(f"Writing configuration to {config_file_path}.")
        f.write(yaml.dump(new_config_file))


def create_config(existing_config: str) -> dict:
    region = click.prompt(
        "🌍 Bedrock region",
        default=existing_config["region"] if existing_config else None,
    )

    available_profiles = click.Choice(boto3.session.Session().available_profiles)
    aws_profile = click.prompt(
        "👤 AWS profile",
        type=available_profiles,
        default=existing_config["aws_profile"] if existing_config else None,
    )

    bedrock = boto3.Session(profile_name=aws_profile).client("bedrock", region)
    all_models = bedrock.list_foundation_models()["modelSummaries"]

    applicable_models = [
        model
        for model in all_models
        if model["outputModalities"] == ["TEXT"]
        and model["inputModalities"] == ["TEXT"]
    ]

    available_models = click.Choice([model["modelId"] for model in applicable_models])
    model_id = click.prompt(
        "🚗 Model",
        type=available_models,
        default=existing_config["model_id"] if existing_config else None,
    )

    model_params = multiline_prompt(
        lambda: click.prompt(
            "🔠 Model params (JSON)",
            default=existing_config["model_params"] if existing_config else "{}",
        ),
        return_newlines=False,
    )
    config = {
        "region": region,
        "aws_profile": aws_profile,
        "model_id": model_id,
        "model_params": model_params,
    }

    llm = model_from_config(config)
    prompt = "Human: You are an assistant used in a CLI tool called 'Ask Bedrock'. The user has just completed their configuration. Write them a nice hello message, including saying that it is from you.\nAssistant:"

    try:
        click.secho(
            llm.predict(prompt),
            fg="yellow",
        )
    except Exception as e:
        if isinstance(e, ValueError) and "AccessDeniedException" in str(e):
            click.secho(
                f"{e}\nAccess denied while trying out the model. Have you enabled model access? Go to the Amazon Bedrock console and select 'Model access' to make sure. Alternatively, choose a different model.",
                fg="red",
            )
            return None
        else:
            click.secho(
                f"{e}\nSomething went wrong while trying out the model, not saving this.",
                fg="red",
            )
            return None

    return config


class YellowStreamingCallbackHandler(StreamingStdOutCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        sys.stdout.write(click.style(token, fg="yellow"))
        sys.stdout.flush()

    def on_llm_end(self, response, **kwargs) -> None:
        sys.stdout.write("\n")
        sys.stdout.flush()


def model_from_config(config: dict) -> Bedrock:
    model_id = config["model_id"]
    credentials_profile_name = config["aws_profile"]
    region = config["region"]
    bedrock = boto3.Session(profile_name=credentials_profile_name).client(
        "bedrock", region
    )
    streaming = bedrock.get_foundation_model(modelIdentifier=model_id)["modelDetails"][
        "responseStreamingSupported"
    ]

    return Bedrock(
        credentials_profile_name=credentials_profile_name,
        model_id=model_id,
        region_name=region,
        streaming=streaming,
        callbacks=[YellowStreamingCallbackHandler()],
        model_kwargs=json.loads(config["model_params"]),
    )


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
