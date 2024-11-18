import datetime
import json
import os
import re
import typing

import dotenv as env
import pandas as pd
import snowflake.connector
import streamlit as st

# Load environment variables from .env file
env.load_dotenv()


class Types:
    QUERY_RESULT = "data"
    ASSISTANT_INIT = "init"
    USER = "user"
    MARKDOWN = "markdown"


class Roles:  # pylint: disable=too-few-public-methods
    USER = "user"
    ASSISTANT = "assistant"


# PROMPT
TIMESPINE_NAME = "date"
PROMPT = None
with open("./prompt.md", encoding="utf8") as s:
    PROMPT = s.read().format(
        today=datetime.datetime.now().strftime("%Y-%m-%d"),
        timespine_name=TIMESPINE_NAME,
    )

SESSION = None
CONECTION = None
try:
    from snowflake.snowpark.context import get_active_session

    SESSION = get_active_session()
    SNOWPARK_AVAILABLE = True
except Exception:
    CONECTION = snowflake.connector.connect(
        user=os.getenv("SF_USER"),
        password=os.getenv("SF_PASSWORD"),
        account=os.getenv("SF_ACCOUNT"),
        host=os.getenv("SF_HOST"),
        port=443,
        warehouse=os.getenv("SF_WAREHOUSE"),
        role=os.getenv("SF_ROLE"),
    )
    CONECTION.cursor().execute(f"USE WAREHOUSE {os.getenv('SF_WAREHOUSE')}")


def escape_quote(str_val: str) -> str:
    str_val = re.sub(r"'+", "''", str_val, flags=re.S)
    return str_val


def run_sql_query(sql: str) -> typing.Any:
    r = None
    if SESSION is not None:
        r = SESSION.execute(sql).fetchall().collect()

    elif CONECTION is not None:
        r = CONECTION.cursor().execute(sql).fetchall()

    return r


def run_cortex(
    sys_prompt: str,
    user_question: str,
    parent: typing.Any,
    temperature: float = 0.2,
) -> typing.Any:

    def extract_pre_json_text(text: str) -> str:
        # Look for markdown JSON code block indicator
        # Handle both ```json and plain ``` markers
        markers = ["```json", "```"]

        first_marker_pos = len(text)
        for marker in markers:
            pos = text.find(marker)
            if pos != -1 and pos < first_marker_pos:
                first_marker_pos = pos

        return text[:first_marker_pos].rstrip()

    data = run_sql_query(
        f"""
    SELECT SNOWFLAKE.CORTEX.COMPLETE('{os.getenv("CORTEX_LLM")}',
    [
        {{'role':'system', 'content': '{escape_quote(sys_prompt)}'}},
        {{'role':'user', 'content': '{escape_quote(user_question)}'}}
    ],
    {{'temperature': {temperature}, 'max_tokens': 500}}) as RESPONSE""",
    )

    cortex_response = json.loads(data[0][0])["choices"][0]["messages"]

    message = extract_pre_json_text(cortex_response)

    append_content(
        parent=parent,
        content={
            "type": Types.MARKDOWN,
            "role": Roles.ASSISTANT,
            "text": f"\n{message}\n",
        },
        arr=st.session_state.content,
    )

    # Parse LLM response to look for Honeydew queries
    for m in re.findall(r"```json\s*\n(.*?)\n\s*```", cortex_response, flags=re.DOTALL):
        return json.loads(m)


def run_honeydew(json_query: typing.Any) -> str:

    def format_as_list(response: typing.Dict[str, typing.Any], key: str) -> str:
        if key not in response:
            return "[]"

        return "[" + ", ".join(f"'{escape_quote(val)}'" for val in response[key]) + "]"

    order_val = json_query.get("order")
    order_fmt = f"'{order_val}'" if order_val is not None else "NULL"
    native_app_sql = f"""
            select {os.getenv('HD_APP')}.API.GET_SQL_FOR_FIELDS(
            '{os.getenv('HD_WORKSPACE')}', '{os.getenv('HD_BRANCH')}', '{os.getenv('HD_DOMAIN')}',
            {format_as_list(json_query, 'group_by')},
            {format_as_list(json_query, 'metrics')},
            {format_as_list(json_query, 'filters')},
            {order_fmt}
        ) AS RESPONSE"""

    sql = run_sql_query(native_app_sql)[0][0]

    # Automatically order by date, if date is part of the response
    if "group_by" in json_query:
        dates = ",".join(
            f'"{val}"'
            for val in json_query["group_by"]
            if val.startswith(f"{TIMESPINE_NAME}.")
        )
        if dates and "order by " not in order_fmt.lower():
            sql += f"\nORDER BY {dates}"

    # Too many rows are not needed
    if "limit " not in order_fmt.lower():
        sql += f"\nLIMIT {os.getenv('RESULT_LIMIT')}"

    return typing.cast(str, sql)


def append_content(
    parent: typing.Any,
    content: typing.Any,
    arr: typing.Optional[typing.List[typing.Any]],
) -> typing.Any:

    if arr is not None:
        arr.append(content)

    if content["type"] == Types.ASSISTANT_INIT:
        parent = parent.chat_message(Roles.ASSISTANT, avatar="🤖")

    if content["type"] == Types.USER:
        parent = parent.chat_message(Roles.USER, avatar="🤓")

    if content["text"] is not None:
        parent.markdown(content["text"])

    if content["type"] == Types.QUERY_RESULT:
        parent.dataframe(content["data"])

    return parent


def process_user_question(user_question: str) -> None:
    append_content(
        parent=st,
        arr=st.session_state.content,
        content={
            "type": Types.USER,
            "role": Roles.USER,
            "text": user_question,
        },
    )
    parent = append_content(
        parent=st,
        arr=st.session_state.content,
        content={
            "type": Types.ASSISTANT_INIT,
            "role": Roles.ASSISTANT,
            "text": "Got it, please wait a sec.",
        },
    )

    json_query = run_cortex(typing.cast(str, PROMPT), user_question, parent)
    if json_query is None:
        return

    append_content(
        parent=parent,
        content={
            "type": Types.MARKDOWN,
            "role": Roles.ASSISTANT,
            "text": f"\n\n```json{json.dumps(json_query, indent=4)}",
        },
        arr=st.session_state.content,
    )

    sql = run_honeydew(json_query)
    if sql is None or sql == "":
        return

    df = pd.DataFrame(run_sql_query(sql))
    append_content(
        parent=parent,
        content={
            "type": Types.QUERY_RESULT,
            "data": df,
            "text": "",
            "role": Roles.ASSISTANT,
        },
        arr=st.session_state.content,
    )


# Main flow

if "content" not in st.session_state:
    st.session_state.content = []

st.title("Honeydew Analyst")
st.markdown(
    f"Semantic Model: `{os.getenv('HD_WORKSPACE')}` on branch `{os.getenv('HD_BRANCH')}`",
)


parent_st = st


for content_item in st.session_state.content:
    if (
        content_item["type"] == Types.ASSISTANT_INIT
        or content_item["type"] == Types.USER
    ):
        parent_st = append_content(parent=st, content=content_item, arr=None)

    else:
        append_content(parent=parent_st, content=content_item, arr=None)

if user_question_input := st.chat_input("Ask me.."):
    process_user_question(user_question_input)
