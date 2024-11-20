from langchain_core.tools import BaseTool
from functools import partial
import logging
import enum
import json
from dataclasses import dataclass
from typing import Optional
import datetime
import time

from langchain_core.messages import SystemMessage, AIMessage, ToolMessage, HumanMessage, BaseMessage
import asyncio
from langgraph.constants import END

from langgraph.checkpoint.memory import MemorySaver

from ai_tools import gen_class_and_return_instance, tools, Tool
import pprint
import os
from langchain_openai import ChatOpenAI
from typing import List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field, ValidationError
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
import logging
import dotenv

dotenv.load_dotenv()

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)

logger = logging.getLogger()


langchain_tool_instances = [
    gen_class_and_return_instance(tool) for tool in tools
]

llm = ChatOpenAI(
    model_name="gpt-4o-2024-08-06",
    temperature=0.1,
    openai_api_key=os.environ['OPENAI_API_KEY'],
    request_timeout=600,
    max_retries=3,
)


class PlanExecuteState(TypedDict):
    input: str
    plan: List[str]
    internal_messages: List[dict]
    response: str
    critics: str


NODE_PLANNER = 'planner'
NODE_AGENT = "agent"
NODE_TOOLS = "tools"
NODE_CRITIC = 'critic'


planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are smart planner agent that can come up with a new plan to solve a problem.

**Plan**
For the given objective, come up with a simple step by step plan.
This plan should involve tasks, that if executed correctly will yield the correct answer. 
Make sure that each step has all the information needed - do not skip steps.
Check the old plan and previous completed steps.
Do not include finished steps in the new plan.

If you have all needed information return final response to the user.

**Final Response**
User does not see any internal messages, he will see only your final response.
So include all relevant information from previous steps to the final message and make it detailed.
""".strip()
        ),
        ("placeholder", "{messages}"),
    ]
)



critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are smart critic.
Check the user request, performed actions and the final response.

Return detailed critics how the result can be improved.
Then it will be forwarded for additional steps.
If you have all needed information, and the result is perfect return final response to the user.

Будь очень критична и попробуй улучшить ответ хотя бы два раза
But do not do it forever.

**Critics**
Should have information how result can be improved.
And what steps could be used
Обязательно попробуй придумать как улучшить ответ или собрать больше инфы или 
подумать еще варианты

**Final Response Indicator**
True if everything is perfect

**Critics**
Sometimes you will see critics for the previous work.
Try to make the work better according to the critics.
And then return final very detailed merged response.
""".strip()
        ),
        ("placeholder", "{messages}"),
    ]
)

class Plan(BaseModel):
    thoughts: str = Field(
        description="Any thoughts/observations on the plan. "
                    "Why it's needed. Which additional step could be performed. "
                    "Which beautification could be done. What should be fixed. "
                    "Can we go deeper into analysis. etc.",
    )
    steps: Optional[List[str]] = Field(
        description="steps to follow, ordered in the way how they should be executed",
        default=None,
    )
    final_response: Optional[str] = Field(
        description="Final response. Just a string.",
        default=None,
    )

    class Config:
        description = "new plan steps or final response"



class Critic(BaseModel):
    thoughts: str = Field(
        description="Any thoughts/observations on the critics. "
                    "Why it's needed. Which additional step could be performed. "
                    "Which beautification could be done. What should be fixed. "
                    "Can we go deeper into analysis. etc.",
    )
    critics: Optional[str] = Field(
        description="what is bad and how to improve, how to make result better",
        default=None,
    )
    final_response_indicator: Optional[bool] = Field(
        description="True if everything is perfect",
        default=None,
    )

    class Config:
        description = "critics or final_response_indicator"


async def _execute_step(
    state: PlanExecuteState,
) -> dict:
    plan = state["plan"]
    plan_str = "\n--\n".join(
        f"Step {i + 1}: {step}" for i, step in enumerate(plan))
    task = plan[0]
    system_message = f"""
You are smart task executor.
You need to execute one step from the plan.

The plan:
----------------
{plan_str}
----------------

The last user message:
----------------
{state["input"]}
----------------

You will get the result of the task you executed after the task is done.

Try to complete the task as best as you can even if it is hard and takes multiple steps.
But do not run forever or do impossible tasks.
In the last message after you do not need to continue the work,
return the final response with the very detailed report.

In the final response, you should include all the information needed to complete the user request.

You will see the conversation history and internal AI / tool calls messages, but do not forget your task.
You are tasked with executing step 1.
Step 1 task:
{task}

Execute Tools one-by-one
"""
    messages: list[BaseMessage] = [
        SystemMessage(content=system_message.strip())
    ] + list(state["internal_messages"]) + [
        SystemMessage(content=system_message)
    ]

    _result: AIMessage = await llm_with_tools.ainvoke(messages)
    logger.info(f"_execute_step:\n{pprint.pformat(_result.content)}")
    return {
        "internal_messages": list(state['internal_messages']) + [_result],
    }


async def _plan_step(state: PlanExecuteState):
    planner = (
            planner_prompt |
            llm.with_structured_output(Plan)
    )
    try:
        plan: Plan = await planner.ainvoke({"messages": [
            ("system", f'In your plan you can use following tools:\n{tools_description}'),
            ("user", f'Input:\n{state["input"]}'),
            ("user", f'Previous Plan:\n{state["plan"]}'),
            ("user", f'Critics:\n{state["critics"]}'),
            ("user", f'Check previous steps below:'),
            ] + state["internal_messages"] + [
            ("user", f'Now return the new plan or give final response'),
        ]})
        logger.debug(f'NodePlanResponse:\n{pprint.pformat(plan)}')
        if plan.final_response:
            return {"response": plan.final_response, 'thoughts': plan.thoughts}
        else:
            return {"plan": plan.steps, 'thoughts': plan.thoughts}
    except Exception as e:
        logger.exception(f"Error planning step: {e}")
        raise


async def _critic_step(state: PlanExecuteState):
    critic = (
            critic_prompt |
            llm.with_structured_output(Critic)
    )
    try:
        _critic_result: Critic = await critic.ainvoke({"messages": [
            ("system", f'In your plan you can use following tools:\n{tools_description}'),
            ("user", f'Input:\n{state["input"]}'),
            ("user", f'Previous Plan:\n{state["plan"]}'),
            ("user", f'Check previous steps below:'),
            ] + state["internal_messages"] + [
            ("user", f'This is the current result:\n{state["response"]}'),
            ("user", f'Now criticise the result or give final response'),
        ]})
        logger.info(f'critic thoughts:\n{_critic_result.thoughts}')
        if _critic_result.final_response_indicator:
            return {'thoughts': _critic_result.thoughts}
        else:
            return {"critics": _critic_result.critics,
                    'thoughts': _critic_result.thoughts,
                    "response": ''}
    except Exception as e:
        logger.exception(f"Error planning step: {e}")
        raise


async def _call_tools(
    state: PlanExecuteState,
) -> dict:
    tools_by_name = {tool.name: tool for tool in langchain_tool_instances}
    outputs = []
    for tool_call in state["internal_messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        logger.info(f"Calling tool {tool.name} with args:\n{pprint.pformat(tool_call)}")
        start_at = time.time()
        tool_result = await tool.ainvoke(tool_call["args"])
        elapsed = time.time() - start_at
        logger.info(f"Tool {tool.name} took: {elapsed:.2f} seconds")
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {
        "internal_messages": list(state['internal_messages']) + outputs,
    }


def should_continue_agent_execution(state: PlanExecuteState):
    internal_messages = state["internal_messages"]
    last_message = internal_messages[-1]
    if not last_message.tool_calls:
        logger.info("No tool calls in the last message. Go to Planner")
        return NODE_PLANNER
    else:
        logger.info("Tool calls in the last message. Go to Agent")
        return NODE_TOOLS



tools_description = "\n--\n".join(
    f"name: {tool.name}:\ndescription: {tool.description}"
    for tool in langchain_tool_instances
)

llm_with_tools = llm.bind_tools(langchain_tool_instances)


workflow = StateGraph(PlanExecuteState)
workflow.add_node(NODE_PLANNER, _plan_step)
workflow.add_node(NODE_AGENT, _execute_step)
workflow.add_node(NODE_TOOLS, _call_tools)
workflow.add_node(NODE_CRITIC, _critic_step)

workflow.add_edge(START, NODE_PLANNER)


def plan_then(state: PlanExecuteState) -> str:
    if ("response" in state and state["response"]):
        return NODE_CRITIC
    else:
        return NODE_AGENT

workflow.add_conditional_edges(
    NODE_PLANNER,
    plan_then,
    {
        NODE_CRITIC: NODE_CRITIC,
        NODE_AGENT: NODE_AGENT,
    }
)

def critic_then(state: PlanExecuteState) -> str:
    if ("response" in state and state["response"]):
        return END
    else:
        return NODE_PLANNER

workflow.add_conditional_edges(
    NODE_CRITIC,
    critic_then,
    {
        END: END,
        NODE_PLANNER: NODE_PLANNER,
    }
)

workflow.add_conditional_edges(
    NODE_AGENT,
    should_continue_agent_execution,
    {
        NODE_PLANNER: NODE_PLANNER,
        NODE_TOOLS: NODE_TOOLS,
    }
)
workflow.add_edge(NODE_TOOLS, NODE_AGENT)
app = workflow.compile()
app.step_timeout = 300
graph = app.get_graph(xray=True)
logger.info('Graph:\n' + graph.draw_ascii())

this_folder = os.path.dirname(os.path.abspath(__file__))
graph.draw_mermaid_png(
    output_file_path=os.path.join(this_folder, 'mas.png'),
)


async def main():
    state = {
        # "input": "What is the weather in the 2nd largest city of France?",
        # "input": "Check last tweets of Vitalik Buterin",
        "input": "изучи погоду в ташкенте",
        "plan": [],
        "internal_messages": [],
        "response": "",
        "critics": "",
    }
    stream = app.astream(state)
    async for step in stream:
        logger.info(f"Step:\n{pprint.pformat(step)}")
        for sub_state in step.values():
            state.update(sub_state)

    logger.info(f"Final response:\n{state.get('response')}")


if __name__ == "__main__":
    asyncio.run(main())
