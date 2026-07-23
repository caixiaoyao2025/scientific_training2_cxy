import os
import re
import types
import sys
import time as _time
import subprocess
import tempfile
import traceback

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["MCP_DATA_ROOT"] = "data"
os.environ["MCP_APP_ROOT"] = "."

_extra_paths = ["/usr/bin", "/usr/local/bin", "/usr/sbin"]
os.environ["PATH"] = ":".join(_extra_paths) + ":" + os.environ.get("PATH", "")

import server as srv
srv.ensure_runtime_directories()
srv.load_and_register_registry()

from biomni.agent import A1
from biomni.tool.support_tools import _persistent_namespace
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# ── Monkey-patch: always combine stdout+stderr so observations are never empty ──
import biomni.utils as _biomni_utils

_orig_run_bash = _biomni_utils.run_bash_script

def _patched_run_bash(script):
    try:
        script = script.strip()
        if not script:
            return "Error: Empty script"
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            if not script.startswith("#!/"):
                f.write("#!/bin/bash\n")
            if "set -e" not in script:
                f.write("set -e\n")
            f.write(script)
            temp_file = f.name
        os.chmod(temp_file, 0o755)
        env = os.environ.copy()
        cwd = os.getcwd()
        result = subprocess.run(
            [temp_file], shell=True, capture_output=True, text=True,
            check=False, env=env, cwd=cwd,
        )
        os.unlink(temp_file)
        output = result.stdout
        if result.stderr:
            output = (output + "\n" + result.stderr).strip() if output else result.stderr
        if not output.strip():
            output = f"Command completed (exit code {result.returncode}). No output."
        if result.returncode != 0:
            output += f"\n[exit code {result.returncode}]"
        return output
    except Exception as e:
        traceback.print_exc()
        return f"Error running Bash script: {str(e)}"

_biomni_utils.run_bash_script = _patched_run_bash
# Also patch the direct import in a1.py so execute() uses our version
import biomni.agent.a1 as _a1
_a1.run_bash_script = _patched_run_bash

api_key = os.environ.get("SILICONFLOW_API_KEY") or "sk-lufftravmhzgdpudfvbvzwrlfyctebizytmtlbynyiohtkij"

agent = A1(
    path="data",
    llm="Qwen/Qwen2.5-72B-Instruct-128K",
    source="Custom",
    base_url="https://api.siliconflow.cn/v1",
    api_key=api_key,
    expected_data_lake_files=[],
)

def make_direct_wrapper(tool_name):
    def wrapper(*args, **kwargs):
        try:
            result = srv.execute_registered_tool(tool_name, kwargs)
            if isinstance(result, list):
                parts = []
                for item in result:
                    if hasattr(item, "text"):
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(f"[Image: {item.mimeType}]")
                    else:
                        parts.append(str(item))
                return "\n".join(parts)
            return str(result)
        except Exception as e:
            return f"Error calling {tool_name}: {e}"
    wrapper.__name__ = tool_name
    wrapper.__doc__ = f"Bio tool: {tool_name}"
    return wrapper

for name in srv.registered_tool_names:
    if not hasattr(agent, '_custom_functions'):
        agent._custom_functions = {}
    agent._custom_functions[name] = make_direct_wrapper(name)

agent.system_prompt = re.sub(
    r"Import file: mcp_servers\.[^\n]+\n=+\n",
    "The following tools are available directly in your namespace:\n",
    agent.system_prompt,
)

agent.system_prompt += (
    "\n\n=== CRITICAL FORMATTING RULE ===\n"
    "EVERY response that contains code to run MUST include an <execute> tag block.\n"
    "Format: <execute>\\n#!bash\\n<your commands>\\n</execute>\n"
    "Or: <execute>\\n#!python\\n<your code>\\n</execute>\n"
    "Do NOT use markdown code blocks (```) for code execution.\n"
    "Always output <execute> tags directly. This is mandatory.\n"
    "=== END RULE ===\n"
)

_orig_llm = agent.llm

class _LLMProxy:
    def __init__(self, llm):
        object.__setattr__(self, "_llm", llm)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_llm"), name)

    def invoke(self, messages, *args, **kwargs):
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = object.__getattribute__(self, "_llm").invoke(messages, *args, **kwargs)
                break
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    wait = min(2 ** (attempt + 2), 60)
                    print(f"Rate limited (attempt {attempt+1}/{max_retries}), waiting {wait}s...")
                    _time.sleep(wait)
                else:
                    raise
        content = response.content if hasattr(response, "content") else str(response)

        if isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, dict):
                    part = block.get("text") or block.get("content") or ""
                    if isinstance(part, str):
                        text_parts.append(part)
            content = "".join(text_parts)

        if not isinstance(content, str):
            return response

        if "<execute>" not in content and "<solution>" not in content:
            code_block_re = re.compile(r'```(?:bash|sh|shell|python|py|r)?\s*\n(.*?)```', re.DOTALL)
            blocks = code_block_re.findall(content)
            if blocks:
                extracted = "\n".join(blocks)
                first_type = re.search(r'```(\w+)', content)
                code_type = "python"
                if first_type and first_type.group(1).lower() in ("bash", "sh", "shell", "cli"):
                    code_type = "bash"
                fixed = content + f"\n\n<execute>\n#!{code_type}\n{extracted}\n</execute>"
                response = AIMessage(content=fixed)

        return response

object.__setattr__(agent, "llm", _LLMProxy(_orig_llm))

mcp_servers_mod = types.ModuleType("mcp_servers")
bio_mcp_mod = types.ModuleType("mcp_servers.bio_mcp")
mcp_servers_mod.bio_mcp = bio_mcp_mod
sys.modules["mcp_servers"] = mcp_servers_mod
sys.modules["mcp_servers.bio_mcp"] = bio_mcp_mod
for name, func in agent._custom_functions.items():
    setattr(bio_mcp_mod, name, func)
    _persistent_namespace[name] = func

# ── Custom Gradio interface with proper left-panel streaming ──
import gradio as gr
from gradio import ChatMessage
from time import time

SUPPORTED_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".pdf")
agent.main_history_copy = []
thread_id = 42

def generate_response(prompt_input, inner_history=None, main_history=None):
    if main_history is None:
        main_history = []
    if inner_history is None:
        inner_history = []

    text_input = prompt_input.get("text", "")
    files = prompt_input.get("files", [])

    agent.main_history_copy += [{"role": "user", "content": text_input}]
    main_history.append(ChatMessage(role="user", content=text_input if text_input else "[Uploaded file]"))
    main_history.append(ChatMessage(role="assistant", content="Working on your request..."))
    yield inner_history, main_history

    for file_info in files:
        fname = os.path.basename(file_info)
        if fname.endswith(".fastq") or fname.endswith(".fq"):
            text_input += f"\n\nUploaded FASTQ file: {file_info}\nRun fastp on this file directly."
        elif fname.endswith(".sam"):
            text_input += f"\n\nUploaded SAM file: {file_info}\nRun samtools flagstat on this file directly."
        elif fname.endswith(".bed"):
            text_input += f"\n\nUploaded BED file: {file_info}\nStore this path for bedtools intersect."
        elif fname.endswith(".gff") or fname.endswith(".gff3"):
            text_input += f"\n\nUploaded GFF file: {file_info}\nUse this with bedtools intersect against the BED file."
        else:
            text_input += f"\n\nUploaded file: {file_info}\nUse this file directly. Do NOT download from the internet."

    agent_messages = []
    for msg in agent.main_history_copy:
        if msg["role"] == "user":
            agent_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            if msg["content"] != "Working on your request...":
                agent_messages.append(AIMessage(content=msg["content"]))
    agent_messages.append(HumanMessage(content=text_input))

    inputs = {"messages": agent_messages, "next_step": None}
    config = {"recursion_limit": 500, "configurable": {"thread_id": thread_id}}

    t = time()
    solution_found = False
    code_execution_messages = []
    step_count = 0

    if agent.use_tool_retriever:
        inner_history.append(ChatMessage(
            role="assistant", content="Retrieving relevant tools...",
        ))
        yield inner_history, main_history
        try:
            selected_resources_names = agent._prepare_resources_for_retrieval(text_input)
            if selected_resources_names:
                agent.update_system_prompt_with_selected_resources(selected_resources_names)
        except Exception as e:
            print(f"Warning: Tool retrieval failed: {e}")
            inner_history.append(ChatMessage(
                role="assistant", content="Tool retrieval unavailable, proceeding with all tools...",
            ))
            yield inner_history, main_history

    for s in agent.app.stream(inputs, stream_mode="values", config=config):
        t_step = time() - t
        message = s["messages"][-1]

        if message.content == text_input:
            t = time()
            continue

        if isinstance(message.content, str):
            tag_positions = []
            for tag in ["<execute>", "<solution>", "<observation>"]:
                pos = message.content.find(tag)
                if pos != -1:
                    tag_positions.append(pos)

            if tag_positions:
                first_tag_pos = min(tag_positions)
                thinking = message.content[:first_tag_pos].strip()
                if thinking:
                    inner_history.append(ChatMessage(
                        role="assistant", content=thinking,
                        metadata={"title": "Thinking", "log": "Agent reasoning"},
                    ))
                    yield inner_history, main_history

            solution_match = re.search(r"<solution>(.*?)</solution>", message.content, re.DOTALL)
            if solution_match and not solution_found:
                solution = solution_match.group(1).strip()
                # Update the left panel: replace placeholder with actual content
                main_history[-1] = ChatMessage(role="assistant", content=solution, metadata={"title": "Answer"})
                agent.main_history_copy += [{"role": "assistant", "content": solution}]
                solution_found = True
                yield inner_history, main_history

            execute_match = re.search(r"<execute>(.*?)</execute>", message.content, re.DOTALL)
            if execute_match:
                step_count += 1
                code = execute_match.group(1).strip()
                language = "python"
                if code.strip().startswith("#!R"):
                    language = "r"
                    code = re.sub(r"^#!R", "", code, count=1).strip()
                elif code.strip().startswith("#!BASH") or code.strip().startswith("#!CLI"):
                    language = "bash"
                    code = re.sub(r"^#!BASH|^#!CLI", "", code, count=1).strip()

                # Extract a brief description for the left panel
                first_line = code.strip().split("\n")[0][:80] if code.strip() else "running code"
                main_history[-1] = ChatMessage(
                    role="assistant",
                    content=f"Step {step_count}: Executing {language} code...\n```\n{first_line}\n```",
                    metadata={"title": f"Step {step_count}"},
                )
                yield inner_history, main_history

                code_msg = ChatMessage(
                    role="assistant",
                    content=f"##### Code:\n```{language}\n{code}\n```",
                    metadata={"title": "Executing code...", "status": "pending", "start_time": t},
                )
                inner_history.append(code_msg)
                code_execution_messages.append(code_msg)
                yield inner_history, main_history

            observation_match = re.search(r"<observation>(.*?)</observation>", message.content, re.DOTALL)
            if observation_match:
                observation = observation_match.group(1).strip()

                if code_execution_messages:
                    code_msg = code_execution_messages[-1]
                    code_msg.metadata.update({"status": "done", "duration": t_step})

                # Update left panel with observation summary
                obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
                main_history[-1] = ChatMessage(
                    role="assistant",
                    content=f"Step {step_count}: Done.\n```\n{obs_preview}\n```",
                    metadata={"title": f"Step {step_count} complete"},
                )
                yield inner_history, main_history

                inner_history.append(ChatMessage(
                    role="assistant",
                    content=f"##### Observation:\n```\n{observation}\n```",
                    metadata={"status": "done", "duration": t_step, "collapsed": True, "collapsible": True},
                ))
                yield inner_history, main_history

                if isinstance(observation, str) and any(ext in observation for ext in SUPPORTED_EXTENSIONS):
                    matches = re.findall(r"(\S+?(?:\.png|\.jpg|\.jpeg|\.gif|\.bmp|\.webp|\.pdf))", observation)
                    valid_matches = [m for m in matches if not m.startswith(("Warning:", "Error:", "'")) and not m.startswith(".")]
                    if valid_matches:
                        inner_history.append(ChatMessage(role="assistant", content="", metadata={"title": "Files"}))
                        for file_path in valid_matches:
                            file_path = file_path.strip("\"'").strip()
                            abs_path = None
                            if os.path.isabs(file_path) and os.path.exists(file_path):
                                abs_path = file_path
                            elif os.path.exists(os.path.join(os.getcwd(), file_path)):
                                abs_path = os.path.join(os.getcwd(), file_path)
                            elif hasattr(agent, "path") and agent.path and os.path.exists(os.path.join(agent.path, file_path)):
                                abs_path = os.path.join(agent.path, file_path)
                            if abs_path:
                                if file_path.lower().endswith(".pdf"):
                                    inner_history.append(ChatMessage(role="assistant", content=f"Found PDF at: {abs_path}", metadata={"title": "PDF File"}))
                                else:
                                    inner_history.append(ChatMessage(role="assistant", content=gr.Image(abs_path), metadata={"title": "Image Preview"}))
                        yield inner_history, main_history

        t = time()

    if not solution_found:
        final_message = s["messages"][-1].content if s["messages"] else ""
        solution_match = re.search(r"<solution>(.*?)</solution>", final_message, re.DOTALL)
        if solution_match:
            solution = solution_match.group(1).strip()
            main_history[-1] = ChatMessage(role="assistant", content=solution, metadata={"title": "Answer"})
            agent.main_history_copy += [{"role": "assistant", "content": solution}]
        else:
            cleaned_content = re.sub(r"<execute>.*?</execute>", "", final_message, flags=re.DOTALL)
            cleaned_content = re.sub(r"<observation>.*?</observation>", "", cleaned_content, flags=re.DOTALL)
            cleaned_content = re.sub(r"\n\s*\n", "\n\n", cleaned_content)
            if cleaned_content.strip():
                main_history[-1] = ChatMessage(role="assistant", content=cleaned_content.strip(), metadata={"title": "Summary"})
                agent.main_history_copy += [{"role": "assistant", "content": cleaned_content.strip()}]
            else:
                main_history[-1] = ChatMessage(role="assistant", content="Task completed. Check the execution log on the right for details.", metadata={"title": "Summary"})
                agent.main_history_copy += [{"role": "assistant", "content": "Task completed."}]

    inner_history.append(ChatMessage(role="assistant", content="Done.", metadata={"title": "Complete"}))
    yield inner_history, main_history


with gr.Blocks(title="Bioinformatics Tool-Discovery Agent") as demo:
    gr.Markdown("# Bioinformatics Tool-Discovery Agent")
    with gr.Row():
        with gr.Column(scale=1):
            main_chatbot = gr.Chatbot(label="Agent", type="messages", height=700, show_copy_button=True)
        with gr.Column(scale=1):
            innerloop_chatbot = gr.Chatbot(label="Execution Log", type="messages", height=700, show_copy_button=True)
    with gr.Row():
        prompt_input = gr.MultimodalTextbox(
            interactive=True, file_count="multiple",
            placeholder="Ask a bioinformatics question or upload a file...",
            show_label=False,
        )
    prompt_input.submit(
        generate_response,
        [prompt_input, innerloop_chatbot, main_chatbot],
        [innerloop_chatbot, main_chatbot],
    ).then(lambda: gr.MultimodalTextbox(value=None), None, [prompt_input])

print("Launching Gradio interface...")
demo.launch(share=True, server_name="0.0.0.0")
