const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
const threadInput = $("#threadId");
const approvalBox = $("#approval");
const activity = $("#activity");
let busy = false;
let activeController = null;

function newThreadId() {
  return `web-${crypto.randomUUID()}`;
}

threadInput.value = localStorage.getItem("agent-thread-id") || newThreadId();

function saveThread() {
  const value = threadInput.value.trim() || newThreadId();
  threadInput.value = value;
  localStorage.setItem("agent-thread-id", value);
  return value;
}

function addMessage(role, text = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const avatar = document.createElement("div");
  avatar.className = "avatar";
  avatar.textContent = role === "user" ? "你" : "A";
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.append(avatar, bubble);
  messages.append(article);
  messages.scrollTop = messages.scrollHeight;
  return bubble;
}

function addActivity(text, error = false) {
  const item = document.createElement("li");
  item.textContent = text;
  if (error) item.className = "error";
  activity.append(item);
  activity.scrollTop = activity.scrollHeight;
}

function setBusy(value, status = "") {
  busy = value;
  $("#send").disabled = value;
  $("#question").disabled = value;
  $("#stop").classList.toggle("hidden", !value);
  if (status) $("#requestStatus").textContent = status;
}

function showResult(data, streamedText = "") {
  $("#requestId").textContent = data.request_id || "—";
  $("#requestStatus").textContent = data.status;
  $("#modelCalls").textContent = data.model_calls ?? 0;
  if (data.answer && !streamedText.includes(data.answer)) {
    addMessage("assistant", data.answer);
  }
  if (data.status === "awaiting_confirmation") {
    approvalBox.classList.remove("hidden");
    const calls = data.pending_action?.tool_calls || [];
    $("#approvalText").textContent = JSON.stringify(calls);
  } else {
    approvalBox.classList.add("hidden");
  }
}

function parseSseBlock(block) {
  let event = "message";
  const data = [];
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    if (line.startsWith("data:")) data.push(line.slice(5).trim());
  }
  return { event, data: JSON.parse(data.join("\n") || "{}") };
}

function describeEvent(message) {
  const data = message.data;
  if (message.event === "started") return `请求开始：${data.operation}`;
  if (message.event === "node_finished") return `节点完成：${data.node}`;
  if (message.event === "tool_requested") return `申请工具：${data.name}`;
  if (message.event === "tool_finished") {
    return `工具${data.success ? "完成" : "失败"}：${data.name}`;
  }
  if (message.event === "approval_required") {
    return `等待确认：${data.tools.join(", ")}`;
  }
  if (message.event === "heartbeat") return "连接心跳";
  if (message.event === "done") return `请求结束：${data.status}`;
  return null;
}

async function consumeStream(url, payload) {
  activeController = new AbortController();
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: activeController.signal,
  });
  if (!response.ok || !response.body) throw new Error("无法连接Agent服务");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const bubbles = new Map();
  let streamedText = "";
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      const message = parseSseBlock(block);
      const description = describeEvent(message);
      if (description) addActivity(description);
      if (message.event === "started") {
        $("#requestId").textContent = message.data.request_id;
        setBusy(true, "Agent运行中");
      }
      if (message.event === "token") {
        const key = message.data.message_id || "current";
        if (!bubbles.has(key)) bubbles.set(key, addMessage("assistant"));
        bubbles.get(key).textContent += message.data.text;
        streamedText += message.data.text;
        messages.scrollTop = messages.scrollHeight;
      }
      if (message.event === "result") showResult(message.data, streamedText);
      if (message.event === "error") {
        addActivity(message.data.message, true);
        throw new Error(message.data.message);
      }
    }
    if (done) break;
  }
}

async function runStream(url, payload, runningText) {
  setBusy(true, runningText);
  try {
    await consumeStream(url, payload);
  } catch (error) {
    if (error.name === "AbortError") {
      addActivity("用户取消请求");
      $("#requestStatus").textContent = "已取消";
    } else {
      addMessage("assistant", `请求失败：${error.message}`);
    }
  } finally {
    activeController = null;
    setBusy(false);
  }
}

async function decide(approved) {
  await runStream(
    "/api/approvals/stream",
    { thread_id: saveThread(), approved },
    approved ? "正在确认" : "正在拒绝",
  );
}

$("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const input = $("#question");
  const question = input.value.trim();
  if (!question) return;
  addMessage("user", question);
  input.value = "";
  await runStream(
    "/api/chat/stream",
    { question, thread_id: saveThread(), max_steps: 6 },
    "正在连接",
  );
});

$("#stop").addEventListener("click", () => activeController?.abort());
$("#approve").addEventListener("click", () => decide(true));
$("#reject").addEventListener("click", () => decide(false));
$("#clearTrace").addEventListener("click", () => { activity.innerHTML = ""; });
$("#newThread").addEventListener("click", () => {
  threadInput.value = newThreadId();
  saveThread();
  location.reload();
});
$("#loadSession").addEventListener("click", async () => {
  try {
    const id = encodeURIComponent(saveThread());
    const response = await fetch(`/api/sessions/${id}`);
    if (!response.ok) throw new Error("会话不存在");
    const session = await response.json();
    messages.innerHTML = "";
    for (const item of session.messages) {
      if (item.type === "human") addMessage("user", item.content);
      if (item.type === "ai" && item.content) addMessage("assistant", item.content);
    }
    $("#requestStatus").textContent =
      `Checkpoint：${session.next_nodes.join(", ") || "完成"}`;
  } catch (error) {
    addMessage("assistant", `读取失败：${error.message}`);
  }
});

fetch("/health")
  .then((response) => response.json())
  .then(() => {
    $("#health").textContent = "服务正常";
    $("#health").classList.add("online");
  })
  .catch(() => { $("#health").textContent = "服务不可用"; });
