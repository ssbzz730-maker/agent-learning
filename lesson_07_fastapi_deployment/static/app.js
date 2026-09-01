const $ = (selector) => document.querySelector(selector);
const messages = $("#messages");
const threadInput = $("#threadId");
const approvalBox = $("#approval");
let busy = false;

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

function addMessage(role, text) {
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
}

function setBusy(value, status = "") {
  busy = value;
  $("#send").disabled = value;
  $("#question").disabled = value;
  if (status) $("#requestStatus").textContent = status;
}

function showResult(data) {
  $("#requestId").textContent = data.request_id || "—";
  $("#requestStatus").textContent = data.status;
  $("#modelCalls").textContent = data.model_calls ?? 0;
  if (data.answer) addMessage("assistant", data.answer);
  if (data.status === "awaiting_confirmation") {
    approvalBox.classList.remove("hidden");
    $("#approvalText").textContent = JSON.stringify(data.pending_action?.tool_calls || []);
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

async function streamChat(question) {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, thread_id: saveThread(), max_steps: 6 }),
  });
  if (!response.ok || !response.body) throw new Error("无法连接Agent服务");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() || "";
    for (const block of blocks) {
      if (!block.trim()) continue;
      const message = parseSseBlock(block);
      if (message.event === "started") setBusy(true, "Agent运行中");
      if (message.event === "result") showResult(message.data);
      if (message.event === "error") throw new Error(message.data.message);
    }
    if (done) break;
  }
}

async function decide(approved) {
  setBusy(true, approved ? "正在确认" : "正在拒绝");
  try {
    const response = await fetch("/api/approvals", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ thread_id: saveThread(), approved }),
    });
    if (!response.ok) throw new Error((await response.json()).detail || "恢复失败");
    showResult(await response.json());
  } catch (error) {
    addMessage("assistant", `错误：${error.message}`);
  } finally {
    setBusy(false);
  }
}

$("#chatForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (busy) return;
  const input = $("#question");
  const question = input.value.trim();
  if (!question) return;
  addMessage("user", question);
  input.value = "";
  setBusy(true, "已发送");
  try {
    await streamChat(question);
  } catch (error) {
    addMessage("assistant", `请求失败：${error.message}`);
  } finally {
    setBusy(false);
  }
});

$("#approve").addEventListener("click", () => decide(true));
$("#reject").addEventListener("click", () => decide(false));
$("#newThread").addEventListener("click", () => {
  threadInput.value = newThreadId();
  saveThread();
  location.reload();
});
$("#loadSession").addEventListener("click", async () => {
  try {
    const response = await fetch(`/api/sessions/${encodeURIComponent(saveThread())}`);
    if (!response.ok) throw new Error("会话不存在");
    const session = await response.json();
    messages.innerHTML = "";
    for (const item of session.messages) {
      if (item.type === "human") addMessage("user", item.content);
      if (item.type === "ai" && item.content) addMessage("assistant", item.content);
    }
    $("#requestStatus").textContent = `Checkpoint：${session.next_nodes.join(", ") || "完成"}`;
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
