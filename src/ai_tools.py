import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, List, Dict, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, create_model

from typing import List, Optional, TypedDict
import pprint
import asyncio
import os
import dotenv

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

SEARCHAPI_API_KEY = os.getenv("SEARCHAPI_API_KEY")
SHORT_TERM_CACHE = 24 * 3600  # todo: replace with 600


@dataclass
class Variable:
    """
    Represents a variable used in the URL or parameters.

    Attributes:
        description (str): A brief description of the variable.
        name (str): The name of the variable.
        constant (bool): Indicates if the variable is a constant value. Defaults to False.
        value (Any): The value of the variable if it's a constant. Defaults to None.
        example (Any): An example value for the variable. Defaults to None.
        required (bool): Indicates if the variable is required. Defaults to True.
        mapping (Optional[Dict[str, str]]): New attribute for mapping. Defaults to None.
    """
    description: str
    name: str
    data_type: type = str
    constant: bool = False  # True if this is a constant value
    value: Any = None  # Value of the variable if it's a constant
    example: Any = None  # Example value for the variable
    required: bool = True  # Indicates if the variable is required
    mapping: Optional[Dict[str, str]] = None  # New attribute for mapping

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Variable":
        return cls(
            description=data["description"],
            name=data["name"],
            constant=data.get("constant", False),
            value=data.get("value", None),
            example=data.get("example", None),
            required=data.get("required", True),
            data_type=data.get("data_type", str),
            mapping=data.get("mapping", None),  # Load mapping from JSON
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "name": self.name,
            "constant": self.constant,
            "value": self.value,
            "example": self.example,
            "required": self.required,
            "data_type": self.data_type,
            "mapping": self.mapping,  # Include mapping in JSON
        }


@dataclass
class Params:
    """Represents the GET/POST parameters for the tool."""
    variables: List[Variable] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Params":
        return cls(
            variables=[Variable.from_json(var) for var in data.get("variables", [])]
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "variables": [var.to_json() for var in self.variables]
        }


@dataclass
class UrlPath:
    """Represents the structure of the URL path with placeholders for variables."""
    path: str
    variables: List[Variable] = field(default_factory=list)

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "UrlPath":
        return cls(
            path=data["path"],
            variables=[Variable.from_json(var) for var in data.get("variables", [])]
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "variables": [var.to_json() for var in self.variables]
        }


@dataclass
class ToolOptions:
    """Defines the options for each API tool, including headers, parameters, and URL."""
    url_path: UrlPath
    http_method: str  # GET or POST
    headers: List[Dict[str, str]] = field(default_factory=list)
    url_get_params: Optional[Params] = None
    url_post_params: Optional[Params] = None
    header_params: Optional[Params] = None

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "ToolOptions":
        return cls(
            url_path=UrlPath.from_json(data["url_path"]),
            http_method=data["http_method"],
            headers=data.get("headers", []),
            url_get_params=Params.from_json(data["url_get_params"]) if data.get("url_get_params") else None,
            url_post_params=Params.from_json(data["url_post_params"]) if data.get("url_post_params") else None,
            header_params=Params.from_json(data["header_params"]) if data.get("header_params") else None,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "url_path": self.url_path.to_json(),
            "http_method": self.http_method,
            "headers": self.headers,
            "url_get_params": self.url_get_params.to_json() if self.url_get_params else None,
            "url_post_params": self.url_post_params.to_json() if self.url_post_params else None,
            "header_params": self.header_params.to_json() if self.header_params else None,
        }


@dataclass
class Tool:
    """Represents the structure of an API tool."""
    name: str
    description: str
    options: ToolOptions
    tool_processor: str = None
    cache_for: Optional[float] = None  # New parameter for caching duration

    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> "Tool":
        return cls(
            name=data["name"],
            description=data["description"],
            options=ToolOptions.from_json(data["options"]),
            tool_processor=data["tool_processor"],
            cache_for=data.get("cache_for", None)  # Load cache_for from JSON
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "options": self.options.to_json(),
            "tool_processor": self.tool_processor,
            "cache_for": self.cache_for  # Include cache_for in JSON
        }


def generate_args_schema(tool: Tool) -> Type[BaseModel]:
    # Replace invalid characters in tool name to create a valid class name
    class_name = re.sub(r'[^a-zA-Z0-9_]', '_', tool.name) + "Args"

    # Create a dictionary of fields for the schema
    fields = {}
    for variable in tool.options.url_path.variables:
        if not variable.constant:
            fields[variable.name] = (Optional[str], None)

    if tool.options.url_get_params:
        for variable in tool.options.url_get_params.variables:
            if not variable.constant:
                fields[variable.name] = (Optional[str], None)

    if tool.options.url_post_params:
        for variable in tool.options.url_post_params.variables:
            if not variable.constant:
                fields[variable.name] = (Optional[variable.data_type], None)

    if tool.options.header_params:
        for variable in tool.options.header_params.variables:
            if not variable.constant:
                fields[variable.name] = (Optional[str], None)

    # Dynamically generate the Pydantic schema model with the fixed class name
    schema_class = create_model(class_name, **fields)
    logger.debug(f"Generated args schema for tool '{tool.name}': {schema_class}")
    return schema_class


disable_tool_cache = True

async def call_tool(tool: Tool, tool_kwargs: dict):
    logger.info(f"ApiCaller: Calling tool: {tool.name}")
    logger.info(f"ApiCaller: tool_kwargs: {tool_kwargs}")

    url = tool.options.url_path.path

    # Gather all non-constant parameters
    non_constant_params = {}

    # Replace variables in the URL and gather non-constant parameters
    for variable in tool.options.url_path.variables:
        if variable.constant:
            value = variable.value
        else:
            value = tool_kwargs.get(variable.name)
            if value is None:
                raise ValueError(f"Missing required variable: {variable.name}")
            non_constant_params[variable.name] = value
        url = url.replace(f"{{{variable.name}}}", str(value))

    # Prepare GET and POST parameters
    get_params = {}
    post_params = {}

    if tool.options.url_get_params:
        for variable in tool.options.url_get_params.variables:
            if variable.constant:
                get_params[variable.name] = variable.value
            else:
                value = tool_kwargs.get(variable.name)
                get_params[variable.name] = value
                non_constant_params[variable.name] = value

    if tool.options.url_post_params:
        for variable in tool.options.url_post_params.variables:
            if variable.constant:
                post_params[variable.name] = variable.value
            else:
                value = tool_kwargs.get(variable.name)
                post_params[variable.name] = value
                non_constant_params[variable.name] = value

    headers = {}
    for _h in tool.options.headers:
        headers.update(_h)

    if tool.options.header_params:
        for variable in tool.options.header_params.variables:
            if variable.constant:
                headers[variable.name] = variable.value
            else:
                value = tool_kwargs.get(variable.name)
                if variable.name == 'bearer':
                    headers["Authorization"] = f"Bearer {value}"
                else:
                    headers[variable.name] = value
                non_constant_params[variable.name] = value

    # Build CURL command for logging
    curl_headers = [f'-H "{k}: {v}"' for k, v in headers.items()]
    curl_headers = (' \\' + "\n").join(curl_headers)
    _url_with_params = url + "?" + "&".join(f"{k}={v}" for k, v in get_params.items())
    curl_command = f"curl -X {tool.options.http_method} {_url_with_params} {curl_headers}"

    # Add POST data if applicable
    if tool.options.http_method == "POST":
        curl_command += "\\\n -H 'Content-Type: application/json'"
        jsonified_data = json.dumps(post_params, indent=4)
        jsonified_data = jsonified_data.replace("'", "\\'")
        curl_command += f"\\\n -d '{jsonified_data}'"

    sep = '='*40
    logger.info(f"ApiCaller: Making CURL request:\n{sep}\n{curl_command}\n{sep}")

    import ssl
    from aiohttp import TCPConnector

    # Создаем SSLContext с отключенной проверкой сертификата
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    max_retries = 4
    for attempt in range(max_retries + 1):
        # Создаем коннектор с отключенной проверкой SSL
        connector = TCPConnector(ssl=ssl_context)
        try:
            # Make the actual API request
            async with aiohttp.ClientSession(headers=headers, timeout=aiohttp.ClientTimeout(total=600), connector=connector) as session:
                async with session.request(
                        url=url,
                        method=tool.options.http_method,
                        params=get_params,
                        json=post_params,
                ) as response:
                    text = await response.text()
                    if response.status == 429:
                        logger.warning(f"Received 429 Too Many Requests. Attempt {attempt + 1} of {max_retries + 1}.")
                        if attempt < max_retries:
                            delay = 2 ** attempt
                            logger.info(f"Waiting {delay} seconds before retrying...")
                            await asyncio.sleep(delay)
                            continue
                        else:
                            logger.error("Max retries reached. Aborting.")
                            return f"error: Received 429 Too Many Requests after retries."
                    elif response.ok:
                        pass  # all good
                    else:
                        logger.info(f"ApiCaller: API call bad response:\n{text}")
                        logging.warning(f"API call failed with status code {response.status}")
                        return (f"error ApiCaller: API call bad response:\n"
                                f"{response.status=}\n"
                                f"{text=}\n"
                                )


                    if tool.name == "fetch-url-rendered-html":
                        logging.warning(f'apply post-processing hack for fetch-url-rendered-html, todo: change the tool pattern')
                        browser_html = json.loads(text)['browserHtml']
                        import html2text
                        h = html2text.HTML2Text(baseurl=tool_kwargs.get('url'))
                        markdown_content = h.handle(browser_html)
                        markdown_content = markdown_content[:10_000]  # todo rough! use better way
                        logger.info(f"ApiCaller: HTML to Markdown conversion:\n{markdown_content[:64]}")
                        text = markdown_content
                    elif tool.name == "fetch_coingecko_token_info_basic":
                        logging.warning(f'apply post-processing hack for fetch_coingecko_token_info_basic, todo: change the tool pattern')
                        len_before = len(text)
                        data = json.loads(text)
                        if "market_data" in data:
                            del data["market_data"]
                        if "tickers" in data:
                            del data["tickers"]
                        text = json.dumps(data)
                        len_after = len(text)
                        logger.warning(f'fetch_coingecko_token_info_basic: text length before: {len_before}, after: {len_after}')
                    elif tool.name == 'twitter_fetch_tweets':
                        logging.warning(f'apply post-processing hack for twitter_fetch_tweets, todo: change the tool pattern')
                        try:
                            data = json.loads(text)
                            if "tweets" in data:
                                tweets = data["tweets"]
                                for tweet in tweets:
                                    _id = tweet['id_str']
                                    if 'link' in tool_kwargs:
                                        tweet['link'] = f"{tool_kwargs['link']}/status/{_id}"
                            text = json.dumps(data, indent=2)
                        except Exception as e:
                            logging.exception(f'Error parsing JSON response: {e}')
                    elif tool.name == "erc20_token_security_audit":
                        logging.warning(f'apply post-processing hack for erc20_token_security_audit, todo: change the tool pattern')
                        try:
                            data = json.loads(text)
                            if "markdown_report" in data:
                                text = data["markdown_report"]
                            else:
                                logging.warning(f'markdown_report field not found in the response')
                        except Exception as e:
                            logging.exception(f'Error parsing JSON response: {e}')

                    logger.info(f'ApiCaller: API call successful:\n{text[:300]}')
                    this_folder = os.path.dirname(os.path.abspath(__file__))
                    tool_output_file_path = os.path.join(this_folder, f'tool_{tool.name}.txt')
                    with open(os.path.join(this_folder, tool_output_file_path), 'w', encoding='utf8') as f:
                        f.write(json.dumps(non_constant_params, indent=4) + '\n')
                        try:
                            f.write(json.dumps(json.loads(text), indent=4))
                        except Exception as e:
                            f.write(text)
                        return text

        except Exception as e:
            return f"error: {e}"

CACHE_VERSION = 'v8'


def gen_class_and_return_instance(_tool: Tool) -> BaseTool:
    """Factory function to generate a LangChain-compatible tool instance dynamically."""

    _args_schema = generate_args_schema(_tool)
    logger.debug(f"Generated args schema for tool '{_tool.name}': {type(_args_schema)}, {_args_schema}")

    class GeneratedTool(BaseTool):
        name = _tool.name
        description = _tool.description
        args_schema: Type[BaseModel] = _args_schema
        tool = _tool

        async def _arun(self, *args, **kwargs):
            """Asynchronous execution of the tool."""
            logging.debug(f"xxx Calling tool: {type(self.tool)} {self.name} with args: {kwargs}")
            tool_kwargs = {
                key: value for key, value in kwargs.items() if value is not None
            }

            # Apply mapping if set
            for variable in (
                (self.tool.options.url_path.variables if self.tool.options.url_path else []) +
                (self.tool.options.url_get_params.variables if self.tool.options.url_get_params else []) +
                (self.tool.options.url_post_params.variables if self.tool.options.url_post_params else []) +
                (self.tool.options.header_params.variables if self.tool.options.header_params else [])
            ):
                if variable.mapping and (variable_value := tool_kwargs.get(variable.name)):
                    if (mapped := str(variable_value)) in variable.mapping:
                        new_value = variable.mapping[mapped]
                        logger.info(f"Mapping {variable_value} to {new_value}")
                        value = new_value
                    elif (mapped := str(variable_value).lower()) in variable.mapping:
                        new_value = variable.mapping[mapped]
                        logger.info(f"Mapping {variable_value} to {new_value}")
                        value = new_value
                    else:
                        value = variable_value
                    tool_kwargs[variable.name] = value

            # Execute the tool processor or call_tool
            if self.tool.tool_processor:
                tool_processor = globals()[self.tool.tool_processor]
                result = await tool_processor(**tool_kwargs)
            else:
                result = await call_tool(self.tool, tool_kwargs)

            return result

        async def _run(self, *args, **kwargs):
            """Synchronous execution (not implemented)."""
            raise NotImplementedError("Sync version not supported")

    # Return an instance of the dynamically generated class
    return GeneratedTool()


fetch_url_rendered_html_tool = Tool(
    name="fetch-url-rendered-html",
    description="Fetch the rendered HTML content from the specified URL",
    options=ToolOptions(
        headers=[{"Authorization": f"Basic {os.getenv('ZYTE_API_KEY')}"}],
        http_method="POST",
        url_post_params=Params(
            variables=[
                Variable(description="URL to fetch the HTML from", name="url"),
                Variable(description="Should this web page be rendered by a browser", name="browserHtml", constant=True, value=True)
            ]
        ),
        url_path=UrlPath(
            path="https://api.zyte.com/v1/extract",
        )
    ),
    cache_for=SHORT_TERM_CACHE  # Cache for 10 minutes
)


twitter_followers_stats_by_username = Tool(
    name="twitter_followers_stats_by_username",
    description="Get statistics on the number of followers by TweetScout categories: influencers, projects, and VC employees. This endpoint receives data in real time, so it may take some time to respond, especially for accounts with a large number of followers.",
    options=ToolOptions(
        headers=[{"ApiKey": os.getenv("TWEETSCOUT_API_KEY")}],
        http_method="GET",
        url_path=UrlPath(
            path="https://api.tweetscout.io/api/followers-stats?username={username}",
            variables=[
                Variable(description="Handler of the Twitter account", name="username", example="tweetscout_io"),
            ]
        )
    ),
    cache_for=SHORT_TERM_CACHE
)

twitter_followers_stats_by_id = Tool(
    name="twitter_followers_stats_by_id",
    description="Get statistics on the number of followers by TweetScout categories: influencers, projects, and VC employees. This endpoint receives data in real time, so it may take some time to respond, especially for accounts with a large number of followers.",
    options=ToolOptions(
        headers=[{"ApiKey": os.getenv("TWEETSCOUT_API_KEY")}],
        http_method="GET",
        url_path=UrlPath(
            path="https://api.tweetscout.io/api/followers-stats",
            variables=[
                Variable(description="ID of the Twitter account", name="id", example="1646549795421421569")
            ]
        )
    ),
    cache_for=SHORT_TERM_CACHE
)


twitter_user_tweets = Tool(
    name="twitter_user_tweets",
    description="Fetch the tweets posted by a specified user. The request requires the user's link and optionally a cursor for pagination. The response includes the tweet texts and a next_cursor for fetching additional tweets if available.",
    options=ToolOptions(
        headers=[{"ApiKey": os.getenv("TWEETSCOUT_API_KEY")}],
        http_method="POST",
        url_post_params=Params(
            variables=[
                Variable(description="Clear user link in Twitter (without ?lang=en, no GET parameters)", name="link", example="https://x.com/tweetscout_io"),
                Variable(description="Cursor to get next 100 tweets", name="cursor", example="DAABCgABGMGwFce__-oKAAIYvwlHghfAYQgAAwAAAAIAAA", required=False),
                Variable(description="ID of the user in Twitter", name="user_id", example="123456789", required=False)
            ]
        ),
        url_path=UrlPath(
            path="https://api.tweetscout.io/api/user-tweets",
        )
    ),
    cache_for=SHORT_TERM_CACHE
)

twitter_account_info = Tool(
    name="twitter_account_info",
    description="Retrieve basic information for a specific Twitter account. The response includes avatar, banner, description, followers count, follows count, account ID, name, registration date, screen name, status count, and verification status.",
    options=ToolOptions(
        headers=[{"ApiKey": os.getenv("TWEETSCOUT_API_KEY")}],
        http_method="GET",
        url_path=UrlPath(
            path="https://api.tweetscout.io/api/info/{username}",
            variables=[
                Variable(description="Username of the Twitter account", name="username", example="tweetscout_io")
            ]
        )
    ),
    cache_for=60*60*4  # Cache for 4 hours
)

user_follows_accounts = Tool(
    name="user_follows_accounts",
    description="Use this tool if you need to find out who a user is following. Get user's follows accounts. Link is a username",
    options=ToolOptions(
        headers=[{"ApiKey": os.getenv("TWEETSCOUT_API_KEY")}],
        http_method="GET",
        url_path=UrlPath(
            path="https://api.tweetscout.io/api/follows-handler",
        ),
        url_get_params=Params(
            variables=[
                Variable(description="Username", name="link", example="tweetscout_io")
            ]
        )
    ),
    cache_for=60*60*4  # Cache for 4 hours
)


async def google_search_processor(query: str) -> str:
    url = 'https://www.searchapi.io/api/v1/search'
    params = {
        'engine': 'google',
        'q': query,
        'api_key': SEARCHAPI_API_KEY
    }
    headers = {
        'Accept': 'application/json'
    }

    def remove_attributes(data):
        if isinstance(data, dict):
            data.pop('related_questions', None)
            data.pop('related_searches', None)
            data.pop('favicon', None)
            data.pop('image', None)
            data.pop('thumbnail', None)
            for key, value in data.items():
                if isinstance(value, str) and len(value) > 3_000:
                    data[key] = value[:100] + '...'
                remove_attributes(value)
        elif isinstance(data, list):
            for item in data:
                remove_attributes(item)

    logging.info(f"google_search_processor: {url=}, {headers=}, {params=}")

    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers, params=params) as response:
            if response.status == 200:
                data = await response.json()
                this_folder = os.path.dirname(os.path.abspath(__file__))
                with open(os.path.join(this_folder, 'google_out_original.json'),
                          'w', encoding='utf8') as f:
                    json.dump(data, f, indent=2)

                remove_attributes(data)

                with open(os.path.join(this_folder, 'google_out_clear.json'),
                          'w', encoding='utf8') as f:
                    json.dump(data, f, indent=2)

                return json.dumps(data, indent=2)
            else:
                return f"Error: {response.status}"


from typing import Dict, List
import json
import aiohttp



google_search_tool = Tool(
    name="google_search_tool",
    description="Perform a Google search using the SearchAPI",
    options=ToolOptions(
        http_method="GET",
        url_get_params=Params(
            variables=[
                Variable(description="Query for the Google search", name="query"),
            ]
        ),
        url_path=UrlPath(
            path="https://www.searchapi.io/api/v1/search",
        )
    ),
    tool_processor='google_search_processor',
    cache_for=24*3600  # Cache for 24 hours
)


web_tools = [
    fetch_url_rendered_html_tool,
    google_search_tool,
]

twitter_read_tools = [
    twitter_followers_stats_by_username,
    twitter_followers_stats_by_id,
    twitter_user_tweets,
    twitter_account_info,
    user_follows_accounts,
]


tools = (
    web_tools +
    twitter_read_tools
)
