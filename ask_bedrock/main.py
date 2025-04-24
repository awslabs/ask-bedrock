from token import NAME
from test.test_traceback import cause_message
import atexit
import asyncio
import json
import logging
import os
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping
from typing import Any, Dict, List, Optional, Tuple
import traceback

import boto3
import click
import yaml
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client

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


def log_error(msg: str, e: Exception | None = None):
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

    if not config:
        log_error("Could not build config")
        return

    start_conversation(config)


@cli.command()
@click.argument("input")
@click.option("-c", "--context", default="default")
@click.option("--debug", is_flag=True, default=False)
def prompt(input: str, context: str, debug: bool):
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)
    config = init_config(context)

    if not config:
        log_error("Could not build config")
        return

    try:
        bedrock_runtime = get_bedrock_runtime(config)
        model_id = config["model_id"]
        inference_config = (
            json.loads(config["inference_config"])
            if "inference_config" in config
            else {}
        )

        # Initialize MCP tools and resources
        tools = []
        resources = []
        if "mcp_servers" in config:
            tools, resources = get_mcp_tools_and_resources(config)
            if tools:
                logger.debug(f"Using {len(tools)} MCP tools")
            if resources:
                logger.debug(f"Using {len(resources)} MCP resources")
    except Exception as e:
        log_error("Error while initializing clients", e)
        return

    try:
        # Format according to the Converse API specification
        messages = [{"role": "user", "content": [{"text": input}]}]
        tool_uses = None
        while tool_uses is None or len(tool_uses) > 0:
            # iterate until no more tools to use
            full_response, tool_uses = stream_response(
                bedrock_runtime,
                model_id,
                messages,
                inference_config,
                tools,
                resources
            )

            messages.append({"role": "assistant", "content": [{"text": full_response}]})
            if len(tool_uses) > 0:
                messages.append({"role": "assistant", "content": [{"toolUse": tool_use} for tool_use in tool_uses]})
                tool_results = use_tools(tool_uses, config)
                messages.append({"role": "user", "content": [{"toolResult": tool_result} for tool_result in tool_results]})

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

        # Initialize MCP tools and resources
        tools = []
        resources = []
        if "mcp_servers" in config:
            tools, resources = get_mcp_tools_and_resources(config)
            if tools:
                logger.debug(f"Using {len(tools)} MCP tools")
            if resources:
                logger.debug(f"Using {len(resources)} MCP resources")
    except Exception as e:
        log_error("Error while initializing clients", e)
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
            tool_uses = None
            while tool_uses is None or len(tool_uses) > 0:
                # iterate until no more tools to use
                full_response, tool_uses = stream_response(
                    bedrock_runtime,
                    model_id,
                    conversation_memory,
                    inference_config,
                    tools,
                    resources
                )

                conversation_memory.append({"role": "assistant", "content": [{"text": full_response}]})
                if len(tool_uses) > 0:
                    conversation_memory.append({"role": "assistant", "content": [{"toolUse": tool_use} for tool_use in tool_uses]})
                    tool_results = use_tools(tool_uses, config)
                    conversation_memory.append({"role": "user", "content": [{"toolResult": tool_result} for tool_result in tool_results]})

        except Exception as e:
            log_error("Error while generating response", e)
            continue

def use_tools(tool_uses: list, config: dict) -> list[dict]:
    tool_results = []
    # Handle tool calls if any occurred
    for tool in tool_uses:
        logger.info(f"Calling tool: {tool['name']}")
        logger.debug(f"Tool call: {tool}")
        try:
            server_name, tool_name = tool['name'].split('___', 1)

            # Find the server config
            server_config = None
            for server in config["mcp_servers"]:
                if server["name"] == server_name:
                    server_config = server
                    break

            if not server_config:
                logger.error(f"Server {server_name} not found in configuration")
                continue

            # Execute the tool call using MCP
            server_params = StdioServerParameters(
                command=server_config["command"],
                args=server_config["args"] if "args" in server_config else [],
                env=server_config["env"] if "env" in server_config else None,
            )

            result = asyncio.run(_call_tool(server_params, tool_name, tool['input']))
            logger.debug(f"Tool result: {result}")

            # Store the tool result for sending back to the model
            tool_results.append({
                "toolUseId": tool["toolUseId"],
                "content": [{
                    "text": "\n".join([text_content.text for text_content in result.content])
                }],
                "status": "error" if result.isError else "success"
            })

        except Exception as e:
            # Add error result for the failed tool call
            tool_results.append({
                "toolUseId": tool["toolUseId"],
                "content": [{
                    "text": str(e)
                }],
                "status": "error"
            })
    return tool_results

def get_config(context: str) -> dict | None:
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
def create_config(existing_config: dict | None) -> dict | None:
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

    # Configure MCP servers
    mcp_servers = []
    if existing_config and "mcp_servers" in existing_config:
        mcp_servers = existing_config["mcp_servers"]

    configure_mcp = click.confirm(
        "📲 Do you want to configure MCP servers?", default=bool(mcp_servers)
    )

    if configure_mcp:
        mcp_servers = configure_mcp_servers(mcp_servers)

    config = {
        "region": region,
        "aws_profile": aws_profile,
        "model_id": model_id,
        "inference_config": inference_config,
    }

    if mcp_servers:
        config["mcp_servers"] = mcp_servers

    # Test the model
    bedrock_runtime = get_bedrock_runtime(config)
    test_message = [
        {
            "role": "user",
            "content": [
                {
                    "text": "I have just completed my configuration. Write me a nice short hello message, "
                    "including saying that it is from you. If there are any tools, summarize their capabilities in two sentences."
                    "Skip confirmation that you understood the request, just do it."
                }
            ],
        }
    ]

    try:
        # Initialize tools and resources for testing
        tools = []
        resources = []
        if "mcp_servers" in config:
            tools, resources = get_mcp_tools_and_resources(config)

        _, _ = stream_response(
            bedrock_runtime,
            model_id,
            test_message,
            json.loads(inference_config),
            tools,
            resources
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


def configure_mcp_servers(existing_servers: List[dict] | None = None) -> List[dict]:
    """Configure MCP servers with user input."""
    if existing_servers is None:
        existing_servers = []

    servers = list(existing_servers)  # Make a copy to avoid modifying the original

    if servers:
        click.echo("Configured MCP servers:")
        for i, server in enumerate(servers):
            click.echo(f"  {i+1}. {server['name']} - {server['command']}")

    while click.confirm("➕ Add a new MCP server?", default=True):
        name = click.prompt("Server name", type=str)
        command = click.prompt("Command to start the server", type=str)

        # Parse command to separate executable from arguments
        cmd_parts = shlex.split(command)
        executable = cmd_parts[0]
        args = cmd_parts[1:] if len(cmd_parts) > 1 else []

        # Optional environment variables
        env_vars = {}
        while click.confirm("Add an environment variable?", default=False):
            env_name = click.prompt("Environment variable name", type=str)
            env_value = click.prompt("Environment variable value", type=str)
            env_vars[env_name] = env_value

        # Test the MCP server connection
        click.echo(f"Discovering tools and resources from {name}...")
        server_config = {
            "name": name,
            "command": executable,
            "args": args,
            "env": env_vars,
        }

        try:
            # Try to connect and discover tools/resources
            tools, resources = discover(server_config)
            click.echo(f"✓ Successfully connected to server '{name}'")
            click.echo(
                f"  - Discovered {len(tools)} tools and {len(resources)} resources"
            )

            # Store the discovered tools and resources in the server config
            server_config["tools"] = tools
            server_config["resources"] = resources

            servers.append(server_config)
        except Exception as e:
            log_error(f"Failed to connect to MCP server: {name}", e)
            if click.confirm("Do you want to add this server anyway?", default=False):
                server_config["tools"] = []
                server_config["resources"] = []
                servers.append(server_config)

    # Ask if user wants to remove any servers
    if len(servers) > 0 and click.confirm(
        "Do you want to remove any MCP servers?", default=False
    ):
        while True:
            for i, server in enumerate(servers):
                click.echo(f"  {i+1}. {server['name']} - {server['command']}")

            server_index = click.prompt(
                "Enter the number of the server to remove (or 0 to finish)",
                type=click.IntRange(0, len(servers)),
                default=0,
            )

            if server_index == 0:
                break

            removed = servers.pop(server_index - 1)
            click.echo(f"Removed server: {removed['name']}")

            if not servers:
                click.echo("No more servers to remove.")
                break

    return servers


async def _call_tool(server_params: StdioServerParameters, tool_name: str, arguments: Dict[str, Any]) -> Any:
    """Call a tool on an MCP server and return the result."""
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
                result = await session.call_tool(tool_name, arguments=arguments)
                return result
            except Exception as e:
                logger.error(f"Error calling tool {tool_name}: {str(e)}")
                raise e

async def _discover(server_config: dict) -> tuple:
    """Connect to an MCP server and discover its tools and resources."""
    server_params = StdioServerParameters(
        command=server_config["command"],
        args=server_config["args"] if "args" in server_config else [],
        env=server_config["env"] if "env" in server_config else None,
    )

    tools = []
    resources = []

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            try:
                await session.initialize()
            except Exception as e:
                log_error(f"Failed to connect to MCP server: {server_config["name"]}", traceback.format_exc())
                raise e
            try:
                    # List available resources
                resources_response = (await session.list_resources()).resources
                resources = [resource.dict() for resource in resources_response]
            except:
                logger.warn(f"Could not list resources for server: {server_config["name"]}. Assuming there is none.")
            try:
                tools_list = (await session.list_tools()).tools
                tools = [tool.dict() for tool in tools_list]
            except:
                logger.warn(f"Could not list tools for server: {server_config["name"]}. Assuming there is none.")


    return tools, resources


def discover(server_config: dict):
    """Test connection to an MCP server and return discovered tools and resources."""
    return asyncio.run(_discover(server_config))


def get_mcp_tools_and_resources(config):
    """Get all MCP tools and resources from configured servers."""
    if "mcp_servers" not in config or not config["mcp_servers"]:
        return [], []

    tools = []
    resources = []

    # Collect tools and resources from all configured servers
    for server in config["mcp_servers"]:
        for tool in server["tools"]:
            tool["server_name"] = server["name"]
            tools.append(tool)
        for resource in server["resources"]:
            resource["server_name"] = server["name"]
            resources.append(resource)

    return tools, resources


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
def init_config(context: str) -> dict | None:
    config = get_config(context)
    if not config:
        click.echo(
            f"No configuration found for context {context}. Creating new configuration."
        )
        config = create_config(None)
        if config:
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


def stream_response(
    bedrock_runtime, model_id, messages, inference_config, tools, resources
) -> tuple[str,list]:

        #TODO decide what to do with resources
    # Prepare request with tools and resources if available
    request_params = {
        "modelId": model_id,
        "messages": messages,
        "system": [{
        "text": "You're an AI assistant for a CLI tool called Ask Bedrock. Depending on the function invoked, the user either chats with you or sends a single prompt. You may or may not have a set of tools at your disposal."
        }],
        "inferenceConfig": inference_config if inference_config else None,
    }
    if tools and len(tools) > 0:
        # map to Converse request schema
        request_params["toolConfig"]= {
            "tools": [{"toolSpec": {
                "description": tool["description"],
                # Prefix tool name with server name to make it uniquely identifiable
                "name": f"{tool['server_name']}___{tool['name']}",
                "inputSchema": {
                    "json": tool["inputSchema"]
                }
            }} for tool in tools]
        }

    try:
        # Execute the request
        logger.debug(f"Sending request to model {model_id}")
        response = bedrock_runtime.converse_stream(**request_params)

        full_response = ""
        tool_uses = []
        current_tool_use = None

        # The response is a stream of events
        for event in response["stream"]:
            if "contentBlockStart" in event:
                current_tool_use = {
                        "toolUseId": event["contentBlockStart"]["start"]["toolUse"]["toolUseId"],
                        "name": event["contentBlockStart"]["start"]["toolUse"]["name"],
                        "input": ""
                    }
            if "contentBlockStop" in event and current_tool_use:
                if current_tool_use:
                    current_tool_use["input"] = json.loads(current_tool_use["input"])
                    tool_uses.append(current_tool_use)
                    current_tool_use = None

            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    chunk_text = delta["text"]
                    full_response += chunk_text
                    sys.stdout.write(click.style(chunk_text, fg="yellow"))
                    sys.stdout.flush()
                if "toolUse" in delta:
                    if not current_tool_use:
                        raise Exception("Tool use block never started")
                    current_tool_use["input"] += delta["toolUse"]["input"]

        sys.stdout.write("\n")
        sys.stdout.flush()

        return full_response, tool_uses
    except Exception as e:
        log_error(f"Error in stream_response: {str(e)}", traceback.format_exc())
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
