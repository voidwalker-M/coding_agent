"""
entry/grpc_server.py

gRPC mirror of the FastAPI task service (JSON payloads via protobuf Struct).

Usage:
  agent serve --grpc-port 50051
"""

from __future__ import annotations

import json
import logging
from concurrent import futures

try:
    import grpc
    from google.protobuf import struct_pb2
except ImportError as exc:
    raise ImportError(
        "gRPC not installed. Run: pip install 'coding-agent[server]'"
    ) from exc

from entry import api as rest

logger = logging.getLogger(__name__)


def _struct_to_dict(msg: struct_pb2.Struct) -> dict:
    from google.protobuf import json_format
    return json_format.MessageToDict(msg)


def _dict_to_struct(data: dict) -> struct_pb2.Struct:
    from google.protobuf import json_format
    s = struct_pb2.Struct()
    json_format.ParseDict(data, s)
    return s


class AgentGrpcServicer:
    """Thin gRPC wrapper over the same in-memory job store as entry.api."""

    def Health(self, request, context):
        return _dict_to_struct(rest.health())

    def SubmitTask(self, request, context):
        body = _struct_to_dict(request)
        from entry.api import TaskSubmitRequest, submit_task
        resp = submit_task(TaskSubmitRequest(**body))
        return _dict_to_struct(resp.model_dump())

    def GetTaskStatus(self, request, context):
        body = _struct_to_dict(request)
        job_id = body.get("job_id")
        if not job_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id required")
        from entry.api import get_task
        resp = get_task(job_id)
        return _dict_to_struct(resp.model_dump())

    def ResumeTask(self, request, context):
        body = _struct_to_dict(request)
        job_id = body.get("job_id")
        if not job_id:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "job_id required")
        from entry.api import ResumeRequest, resume_task
        ckpt = body.get("checkpoint_path")
        resp = resume_task(job_id, ResumeRequest(checkpoint_path=ckpt) if ckpt else None)
        return _dict_to_struct(resp.model_dump())


def _load_servicer_add():
    from entry.proto import agent_pb2_grpc
    return agent_pb2_grpc


def serve_grpc(host: str = "0.0.0.0", port: int = 50051, workers: int = 4) -> None:
    agent_pb2_grpc = _load_servicer_add()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=workers))
    agent_pb2_grpc.add_AgentServiceServicer_to_server(AgentGrpcServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    logger.info("gRPC AgentService listening on %s:%d", host, port)
    server.wait_for_termination()
