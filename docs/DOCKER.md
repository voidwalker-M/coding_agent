# Docker 里跑起来（Postgres + Redis + API）

记忆落到 **Postgres**，缓存走 **Redis**。HTTP API 在 `http://localhost:8766`。不配大模型密钥也能验证数据库和缓存；要让 Agent 改代码才需要密钥。

## 你需要准备的

1. **Docker Desktop**（macOS）或 Docker Engine + Compose v2。本机执行 `docker compose version` 能出版本号即可。
2. 端口空闲：`8766`（API）、`5432`（Postgres）、`6379`（Redis）。
3. **可选** LLM 密钥（任选一个），只在调用 `POST /v1/tasks` 时需要：
   - `GPT_OSS_API_KEY`（默认 config 走 UF LiteLLM）
   - 或 `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `DEEPSEEK_API_KEY`

不配密钥时 `/health` 里 `llm.backend` 会是 `offline`，记忆和缓存接口仍然可用。

## 启动

在仓库根目录：

```bash
docker compose up --build
```

第一次会拉 `postgres:16`、`redis:7` 并构建 Agent 镜像，大约几分钟。看到类似：

```
Agent API listening on http://0.0.0.0:8766
  llm=offline (...)  memory=True
```

就表示栈已经起来。另开一个终端做下面的检查。

## 验证数据库和缓存

```bash
# 健康检查：memory.backend 应为 postgres，cache.backend 应为 redis
curl -s http://localhost:8766/health | python3 -m json.tool

# 写一条长期记忆（进 Postgres）
curl -s -X POST http://localhost:8766/v1/memory \
  -H 'Content-Type: application/json' \
  -d '{"text":"这个仓库用 pytest 跑测试","user_id":"demo"}'

# 按关键词召回
curl -s 'http://localhost:8766/v1/memory?q=pytest&user_id=demo'

# Redis + Postgres 双写缓存
curl -s -X POST http://localhost:8766/v1/cache \
  -H 'Content-Type: application/json' \
  -d '{"namespace":"demo","key":"hello","value":"from-redis"}'

curl -s 'http://localhost:8766/v1/cache?namespace=demo&key=hello'
```

会话短记忆（STM turns）也会进 Postgres：

```bash
CID=$(curl -s -X POST http://localhost:8766/v1/conversations \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo","title":"试跑"}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["id"])')

curl -s -X POST "http://localhost:8766/v1/conversations/$CID/turns" \
  -H 'Content-Type: application/json' \
  -d '{"role":"user","content":"数据库连上了吗","query_index":0}'

curl -s "http://localhost:8766/v1/conversations/$CID/turns"
```

## 可选：带密钥跑一次真正的 Agent 任务

```bash
export GPT_OSS_API_KEY=你的密钥
docker compose up --build
curl -s -X POST http://localhost:8766/v1/tasks \
  -H 'Content-Type: application/json' \
  -d '{"description":"列出仓库根目录有哪些 Python 包","repo_path":"/workspace"}'
```

用返回的 `job_id` 轮询 `GET /v1/tasks/{job_id}`。

## 本机不用 Docker、只连容器里的库

Compose 把 Postgres/Redis 也映射到了本机端口。先 `docker compose up postgres redis`，然后：

```bash
pip install -e ".[server]"
export MEMORY_ENABLED=true
export MEMORY_DATABASE_URL=postgresql://agent:agent@127.0.0.1:5432/agent
export REDIS_URL=redis://127.0.0.1:6379/0
agent serve --host 127.0.0.1 --port 8766 --repo .
```

不设这些环境变量时，记忆仍然默认用仓库下的 SQLite（`.agent_memory/memory.db`），缓存用进程内 dict。

## 停掉

`Ctrl+C` 之后如需清数据：`docker compose down -v`（会删掉 Postgres 卷）。
