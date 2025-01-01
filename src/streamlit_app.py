####################################################################################
# MIT License
#
# Copyright (c) Honeydew Data Inc.

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
####################################################################################

import collections
import enum
import json
import os
import re
import typing
import uuid

import altair as alt
import pandas as pd
import snowflake.connector
import streamlit as st

# Constants

#
# Honeydew related settings
HD_APP = "SEMANTIC_LAYER_ENTERPRISE_EDITION"  # Name of installed Native Application
HD_WORKSPACE = "tpch_demo"  # Honeydew model
HD_BRANCH = "prod"  # Honeydew branch
HD_DOMAIN = "llm"  # Honeydew domain

#
# Behavior & UI  settings
HISTORY_ENABLED = True  # Display chat history
SHOW_SQL_QUERY = 1  # Display SQL tab
SHOW_EXPLAIN_QUERY = 1  # Display Explain tab

# Set to '1' to see debug messages
_DEBUG = 1

# Cortex LLM
# CORTEX_LLM = "llama3.1-405b"
CORTEX_LLM = "mistral-large2"

# Results limit
RESULTS_LIMIT = 10000


HONEYDEW_ICON_URL = "https://honeydew.ai/wp-content/uploads/2022/12/Logo_Icon@2x.svg"

_MAX_VALUES_TO_SHOW = 50
_MAX_VALUES_TO_USE_FOR_FOLLOWUP_QUESTION = 50
_DONT_SHOW_NULLS_IN_DIMENSIONS = True


def supress_failures(
    func: typing.Callable[..., typing.Any],
) -> typing.Callable[..., typing.Any]:
    def func_without_errors(*args: typing.Any, **kw: typing.Any) -> typing.Any:
        try:
            return func(*args, **kw)
        except Exception as exp:  # pylint: disable=broad-exception-caught
            if _DEBUG:
                raise exp

            return None

    return func_without_errors


#
# HD API


def init() -> None:

    if (
        "connection_initialized" in st.session_state
        and st.session_state.connection_initialized is True
    ):
        return

    try:
        from snowflake.snowpark.context import (  # pylint: disable=import-outside-toplevel
            get_active_session,
        )

        st.session_state.SESSION = get_active_session()

    except Exception:  # pylint: disable=broad-exception-caught
        # Load environment variables from .env file
        import dotenv as env  # pylint: disable=import-outside-toplevel

        env.load_dotenv()

        st.session_state.CONNECTION = snowflake.connector.connect(
            user=os.getenv("SF_USER"),
            password=os.getenv("SF_PASSWORD"),
            account=os.getenv("SF_ACCOUNT"),
            host=os.getenv("SF_HOST"),
            port=443,
            warehouse=os.getenv("SF_WAREHOUSE"),
            role=os.getenv("SF_ROLE"),
        )
        st.session_state.CONNECTION.cursor().execute(
            f"USE WAREHOUSE {os.getenv('SF_WAREHOUSE')}",
        )

    st.session_state.connection_initialized = True


def encode(str_val: str) -> str:
    str_val = re.sub(r"'+", "''", str_val, flags=re.S)
    return str_val


def execute_sql(sql: str) -> typing.Any:
    assert sql is not None and len(sql) > 0, "SQL is empty"

    if "SESSION" in st.session_state:
        res = st.session_state.SESSION.sql(sql).collect()

    else:
        assert "CONNECTION" in st.session_state, "No session / connection"
        res = st.session_state.CONNECTION.cursor().execute(sql).fetchall()

    return res


def execute_sql_table(sql: str) -> pd.DataFrame:
    assert sql is not None and len(sql) > 0, "SQL is empty"

    if "SESSION" in st.session_state:
        df = st.session_state.SESSION.sql(sql).to_pandas()
    else:
        assert "CONNECTION" in st.session_state, "No session / connection"
        df = st.session_state.CONNECTION.cursor().execute(sql).fetch_pandas_all()

    # Numbering starts from 1, not 0 for presentation purposes
    df.index = df.index + 1
    return df


def execute_llm(
    messages: typing.List[str],
    temperature: float = 0.2,
    max_tokens: float = 8192,
) -> typing.Any:

    def extract_message(t: str) -> str:
        markers = ["JSON Query:", "```json", "```"]
        first_marker_pos = len(t)
        for marker in markers:
            pos = t.find(marker)
            if pos != -1 and pos < first_marker_pos:
                first_marker_pos = pos

        return t[:first_marker_pos].rstrip()

    def extract_json_query(t: str) -> typing.Any:
        for m in re.findall(r"```json(.*?)```", t, flags=re.DOTALL):
            return json.loads(m)

    # Load environment variables from .env file
    import dotenv as env  # pylint: disable=import-outside-toplevel

    env.load_dotenv()

    query = f"""
        SELECT SNOWFLAKE.CORTEX.COMPLETE('{os.getenv("CORTEX_LLM")}',
        [
            {', '.join(messages)}
        ],
        {{'temperature': {temperature}, 'max_tokens': {max_tokens}}}) as RESPONSE"""

    cortex_response = json.loads(execute_sql(query)[0][0])["choices"][0]["messages"]

    return extract_message(cortex_response), extract_json_query(cortex_response)


def execute_hd_ask_question(
    question: str,
    hist: typing.List[str],
) -> typing.Any:

    sql = f"""select {HD_APP}.API.ASK_QUESTION(
                workspace => '{HD_WORKSPACE}',
                branch => '{HD_BRANCH}',
                question => '{encode(question)}',
                history => { "[" + ",".join(hist) + "]"},
                domain => '{HD_DOMAIN}',
                cortex_llm_name => '{CORTEX_LLM}') as response"""

    r = json.loads(execute_sql(sql)[0][0])
    return r


init()


#
# Chat History


class HistoryItemTypes(enum.Enum):  # pylint: disable=too-few-public-methods
    ASSISTANT_INIT = "assistant_init"
    ASSISTANT_MESSAGE = "assistant_message"
    ERROR = "assistant_error"
    LLM_RESPONSE = "assistant_llm_response"
    QUERY_RESULT = "assistant_query_result"
    USER_MESSAGE = "user"


class Roles(enum.Enum):  # pylint: disable=too-few-public-methods
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class History:
    messages: typing.List[typing.Any]
    ui_model: typing.List[typing.Dict[str, typing.Any]]

    def __init__(
        self,
        name: typing.Optional[str] = None,
        system_prompt: typing.Optional[str] = None,
    ):
        self.system_prompt = system_prompt
        self.name = name
        self.clear()

    def clear(self) -> None:
        self.messages = []
        self.ui_model = []
        if self.system_prompt is not None and self.system_prompt != "":
            self.push_system_message(encode(self.system_prompt))

    def push_system_message(self, msg: str) -> None:
        if msg is not None and msg != "":
            self.messages.append(
                f"""{{'role':'{Roles.SYSTEM.value}', 'content': '{msg}'}}""",
            )

    def to_llm_message(self, msg: typing.Dict[str, typing.Any]) -> str:
        return f"""{{'role':'{msg["role"]}', 'content': '{encode(msg['text'])}'}}"""

    def push_assistant_init(self) -> typing.Dict[str, typing.Any]:
        msg = {
            "type": HistoryItemTypes.ASSISTANT_INIT.value,
            "role": Roles.ASSISTANT.value,
            "text": "Got it, please wait a sec.",
        }

        self.ui_model.append(msg)
        return msg

    def push_assistant_error(
        self,
        error: str,
    ) -> typing.Dict[str, typing.Any]:

        msg = {
            "type": HistoryItemTypes.ERROR.value,
            "role": Roles.ASSISTANT.value,
            "text": f"\n{error}\n",
        }

        # self.messages.append(self.to_llm_message(m))
        self.ui_model.append(msg)

        return msg

    def push_assistant_llm_response(
        self,
        message: str,
        json_query: typing.Optional[typing.Any] = None,
        append_json_to_ui: bool = True,
    ) -> typing.Dict[str, typing.Any]:
        msg = {
            "type": HistoryItemTypes.LLM_RESPONSE.value,
            "role": Roles.ASSISTANT.value,
            "json_query": json_query,
            "text": f"""\n{message}\n```json\n{json.dumps(json_query)}\n```""",
        }

        self.messages.append(self.to_llm_message(msg))

        if not append_json_to_ui:
            msg["text"] = f"\n{message}\n"

        self.ui_model.append(msg)

        return msg

    def push_assistant_query_result(
        self,
        df: pd.DataFrame,
    ) -> typing.Dict[str, typing.Any]:
        key = str(uuid.uuid4())[:8]
        msg = {
            "type": HistoryItemTypes.QUERY_RESULT.value,
            "role": Roles.ASSISTANT.value,
            "data": df,
            "key": key,
            "text": "",
        }

        if len(df) < _MAX_VALUES_TO_USE_FOR_FOLLOWUP_QUESTION:
            json_data = {
                "headers": df.columns.tolist(),
                "data": df.values.tolist(),
            }

            # Wrap it in a Markdown code block
            self.messages.append(
                self.to_llm_message(
                    {
                        "role": Roles.USER.value,
                        "text": "show data",
                    },
                ),
            )
            self.messages.append(
                self.to_llm_message(
                    {
                        "role": Roles.ASSISTANT.value,
                        "text": (
                            f"# JSON DataTable Key = {key}\n"
                            f"```json\n{json.dumps(json_data, default=str)}\n```"
                        ),
                    },
                ),
            )

        self.ui_model.append(msg)

        return msg

    def push_user_message(self, message: str) -> typing.Dict[str, typing.Any]:
        msg = {
            "type": HistoryItemTypes.USER_MESSAGE.value,
            "role": Roles.USER.value,
            "text": message,
        }

        self.messages.append(self.to_llm_message(msg))
        self.ui_model.append(msg)

        return msg


# # Application

if "history" not in st.session_state:
    st.session_state.history = History()

history: History = st.session_state.history


def render_grouped_bar_chart(
    df: pd.DataFrame,
    str_columns: typing.List[str],
    numeric_columns: typing.List[str],
) -> bool:
    if len(str_columns) > 2:
        return False

    params = {}
    if len(str_columns) == 2:

        params["x"] = alt.X(f"{str_columns[0]}:N", title=None)
        params["color"] = (
            "Series:N" if len(numeric_columns) > 1 else f"{str_columns[1]}:N"
        )

        if len(numeric_columns) > 1:
            params["column"] = f"{str_columns[1]}:N"

    else:
        params["x"] = alt.X(f"{str_columns[0]}:N", title=None)
        params["color"] = (
            "Series:N" if len(numeric_columns) > 1 else f"{str_columns[0]}:N"
        )

    if len(numeric_columns) == 1:
        params["y"] = alt.Y("Value:Q", title=numeric_columns[0])

    else:
        params["y"] = alt.Y("Value:Q")
        params["xOffset"] = alt.XOffset("Series:N")

    # Melt the DataFrame to convert wide format into long format for grouped bars
    df = df.melt(
        id_vars=str_columns,
        value_vars=numeric_columns,
        var_name="Series",
        value_name="Value",
    )

    # Create the Altair grouped bar chart using the index as the x-axis
    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(**params)
        .resolve_scale(y="shared")  # Ensure that y-axes across columns are shared
    )

    st.altair_chart(chart, use_container_width=True)

    return True


def _fix_column_name_for_chart(col: str) -> str:
    return col.replace(".", "_")


def _get_columns_and_types(
    json_query: typing.Dict[str, typing.Any],
    section: str,
) -> typing.OrderedDict[str, str]:
    columns = [
        _fix_column_name_for_chart(typing.cast(str, col)) for col in json_query[section]
    ]
    types = [
        typing.cast(str, comp["datatype"]) for comp in json_query["components"][section]
    ]
    return collections.OrderedDict(zip(columns, types))


def render_chart(df: pd.DataFrame, json_query: typing.Dict[str, typing.Any]) -> bool:
    # pylint: disable=too-many-branches,too-many-return-statements

    if len(df) == 0:
        return False

    metrics = _get_columns_and_types(json_query, "metrics")
    attributes = _get_columns_and_types(json_query, "attributes")

    if len(metrics) == 0:
        # No chart when there are no metrics to show
        return False

    # Bug in streamlit with dots in column names - update column names
    updated_columns = {col: _fix_column_name_for_chart(col) for col in df.columns}
    df = df.rename(columns=updated_columns)

    if set(metrics.values()) - {"number", "float"}:
        st.markdown("Chart currently not available for non-numeric metrics")
        return False

    # Bug in snowflake connector - does not set date types correctly in pandas, so manually detecting
    date_columns = [
        col for col, col_type in attributes.items() if col_type in ("date", "timestamp")
    ]
    dim_columns = [col for col in attributes if col not in date_columns]
    metric_columns = list(metrics.keys())
    df_to_show = df[metric_columns]

    for col in date_columns:
        # Fix datatype - as Snowflake returns dates as objects
        df_to_show[col] = pd.to_datetime(df[col], errors="coerce")

    for col in dim_columns:
        # Show all dimensions as strings
        df_to_show[col] = df[col].astype(str)

    # date dimension
    if len(date_columns) > 0:
        df_to_show = df_to_show.rename(columns={date_columns[0]: "index"}).set_index(
            "index",
        )

        if len(dim_columns) == 0:
            st.line_chart(df_to_show)
        elif len(dim_columns) == 1 and len(metric_columns) == 1:
            st.line_chart(df_to_show, y=metric_columns[0], color=dim_columns[0])
        else:
            return False

    # dimensions
    elif len(dim_columns) >= 1:
        metric_column = metric_columns[0]  # First numeric column

        if len(df_to_show) > _MAX_VALUES_TO_SHOW:
            st.markdown(
                f"**Showing top {_MAX_VALUES_TO_SHOW} values sorted by {metric_column}**",
            )
            df_to_show = df_to_show.sort_values(by=metric_column, ascending=False).head(
                _MAX_VALUES_TO_SHOW,
            )

        if len(dim_columns) == 1 and len(metric_columns) == 1:
            chart = (
                alt.Chart(df_to_show)
                .mark_bar()
                .encode(
                    y=alt.X(
                        dim_columns[0],
                        sort=alt.EncodingSortField(
                            field=metric_column,
                            order="descending",
                        ),
                    ),
                    x=alt.Y(metric_columns[0]),
                    color=alt.Color(dim_columns[0], legend=None),  # Hide legend
                )
            )

            st.altair_chart(chart, use_container_width=True)

        else:
            return render_grouped_bar_chart(df_to_show, dim_columns, metric_columns)

    # just metrics
    elif len(dim_columns) == 0 and len(date_columns) == 0:
        if len(df_to_show) == 1:
            for metric_column in metric_columns:
                value = df_to_show[metric_column].iloc[0]
                value_str = "No data" if pd.isna(value) else f"{value:,}"
                st.metric(label=metric_column, value=value_str)
        else:
            # can only happen with partial metrics and no grouping given
            return False

    else:
        st.markdown("Chart currently not available")
        return False

    return True


def clean_dataframe_for_presentation(
    df: pd.DataFrame,
    json_query: typing.Dict[str, typing.Any],
) -> pd.DataFrame:

    attributes = json_query["attributes"]

    # Remove NULLs in dimensions
    if _DONT_SHOW_NULLS_IN_DIMENSIONS and attributes:
        df = df.dropna(subset=attributes)
    else:
        df[attributes] = df[attributes].fillna("No data")

    return df


def render_dataframe(df: pd.DataFrame) -> None:
    if len(df) == 0:
        st.markdown("No data available to display. Please refine your question.")
    st.dataframe(df.fillna("No data"))
    if len(df) > 0:
        csv = typing.cast(bytes, df.to_csv().encode("utf-8"))
        st.download_button(
            label="Download data as CSV",
            data=csv,
            key=f"download_btn_{str(uuid.uuid4())}",
            file_name="data.csv",
            mime="text/csv",
        )


@supress_failures
def render_message(
    message: typing.Dict[str, typing.Any],
    parent: typing.Any,
) -> typing.Optional[typing.Any]:
    # pylint: disable=too-many-statements

    def find_item(
        json_data: typing.Any,
        search_str: str,
    ) -> typing.Optional[typing.Any]:
        # Split the search string into entity and name components
        entity, name = search_str.split(".")

        # Search in both attributes and metrics within components
        for section in ["attributes", "metrics"]:
            for item in json_data.get("components", {}).get(section, []):
                if item.get("entity") == entity and item.get("name") == name:
                    return item
        return None

    def get_order_item_caption(
        json_data: typing.Any,
        name: str,
        order: str,
    ) -> typing.Any:
        parts = [" * "]

        name = name.strip('"')
        order = "ascending" if order == "asc" else "descending"

        item = find_item(json_data, name)

        if item is not None and item["ui_url"] is not None and item["ui_url"] != "":
            parts.append(f"[{name}]({item['ui_url']}) ({order})")
        else:
            parts.append(f"`{name}` ({order})")

        return "".join(parts)

    def get_item_caption(json_data: typing.Any, p: str, name: str) -> typing.Any:
        parts = [" * "]

        item = find_item(json_data, name)

        if item is not None and item.get("ui_url") is not None and item["ui_url"] != "":
            parts.append(f" [{name}]({item['ui_url']})")
        elif p == "filters":
            parts.append(f"`{name}`")
        else:
            parts.append(name)

        if (
            item is not None
            and item["description"] is not None
            and item["description"] != ""
        ):
            parts.append(f"\n\n    {item['description']}")

        return "".join(parts)

    def get_panel_description(
        question: typing.Dict[str, typing.Any],
        p: str,
        n: str,
    ) -> str:
        if (
            p not in question
            or question[p] is None
            or (isinstance(question[p], list) and len(question[p]) == 0)
        ):
            return ""

        if isinstance(question[p], int):
            return f"\n\n**{n}:** {question[p]}\n\n"

        r = f"\n\n**{n}:**\n\n"
        if isinstance(question[p], str):
            r += question[p]
        else:
            if p == "order":
                r += "\n\n".join(
                    get_order_item_caption(question, val["alias"], val["order"])
                    for val in question[p]
                )
            else:
                r += "\n\n".join(
                    get_item_caption(question, p, val) for val in question[p]
                )

        return r

    if message["type"] == HistoryItemTypes.ASSISTANT_INIT.value:
        parent = parent.chat_message(
            Roles.ASSISTANT.value,
            avatar=HONEYDEW_ICON_URL,
        )

    if message["type"] == HistoryItemTypes.USER_MESSAGE.value:
        parent = parent.chat_message(Roles.USER.value, avatar="🧑‍💻")

    if message["text"] is not None:
        parent.markdown(message["text"])

    if message["type"] == HistoryItemTypes.QUERY_RESULT.value:
        df = message["data"]
        json_query = message["json_query"]
        sql_query = message["sql_query"]

        tab_names = ["Chart", "Data"]
        if SHOW_EXPLAIN_QUERY:
            tab_names.append("Explain")

        if SHOW_SQL_QUERY:
            tab_names.append("SQL")

        df = clean_dataframe_for_presentation(df, json_query)
        chart_tab, data_tab, explain_tab, hd_sql_tab = parent.tabs(tab_names)

        with chart_tab:
            if not render_chart(df, json_query):
                render_dataframe(df)

        with data_tab:
            render_dataframe(df)

        if SHOW_EXPLAIN_QUERY:
            with explain_tab:
                st.markdown(
                    "".join(
                        [
                            get_panel_description(
                                json_query,
                                "attributes",
                                "Attributes",
                            ),
                            get_panel_description(json_query, "metrics", "Metrics"),
                            get_panel_description(json_query, "filters", "Filters"),
                            get_panel_description(json_query, "order", "Order"),
                            get_panel_description(json_query, "limit", "Limit"),
                        ],
                    ),
                )

        if SHOW_SQL_QUERY:
            with hd_sql_tab:
                st.markdown(f"```sql{sql_query}\n```")
    return parent


@supress_failures
def process_user_question(user_question: str) -> None:

    history_clone = history.messages[:]
    stat = None
    try:
        render_message(history.push_user_message(user_question), st)
        parent = render_message(history.push_assistant_init(), st)

        container = parent.container()
        stat_parent = parent.empty()
        stat = stat_parent.status(label="Running..", state="running")

        # executing Honeydew
        hdresponse = execute_hd_ask_question(
            user_question,
            hist=history_clone,
        )

        render_message(
            history.push_assistant_llm_response(
                message=hdresponse["llm_response"],
                json_query=(
                    json.loads(hdresponse["llm_response_json"])
                    if hdresponse.get("llm_response_json")
                    else None
                ),
                append_json_to_ui=False,
            ),
            container,
        )

        if "error" in hdresponse and hdresponse["error"] is not None:
            render_message(history.push_assistant_error(hdresponse["error"]), container)
            return

        if "sql" in hdresponse and hdresponse["sql"] is not None:
            df = execute_sql_table(hdresponse["sql"])
            df = df.round(2)
            msg = history.push_assistant_query_result(df)
            msg["json_query"] = hdresponse["perspective"]
            msg["sql_query"] = hdresponse["sql"]
            render_message(msg, container)

            parent.divider()

    finally:
        if stat is not None:
            stat.update(label="Completed", state="complete")
            stat_parent.empty()


#
# Main flow
col1, col2 = st.columns([1, 0.1])
with col1:
    st.title("Honeydew Analyst")
    st.markdown(
        f"Semantic Model: `{HD_WORKSPACE}` on branch `{HD_BRANCH}`",
    )
with col2:
    with st.container():
        # poor man's vertical alignment
        st.write("")
        st.write("")
        if st.button("♻️"):
            history.clear()


parent_st = st

for history_item in history.ui_model:
    if history_item["type"] in (
        HistoryItemTypes.ASSISTANT_INIT.value,
        HistoryItemTypes.USER_MESSAGE.value,
    ):
        parent_st = render_message(history_item, st)
    else:
        render_message(history_item, parent_st)

if user_question_input := st.chat_input(placeholder="Ask me.."):
    process_user_question(user_question_input)
