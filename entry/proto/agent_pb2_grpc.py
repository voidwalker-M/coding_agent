# Generated-style gRPC stubs for agent.proto (Struct in/out).
# Regenerate: python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. entry/proto/agent.proto

from google.protobuf import struct_pb2 as _struct_pb2
import grpc as _grpc


class AgentServiceStub:
    def __init__(self, channel):
        self.Health = channel.unary_unary(
            "/agent.AgentService/Health",
            request_serializer=_struct_pb2.Struct.SerializeToString,
            response_deserializer=_struct_pb2.Struct.FromString,
        )
        self.SubmitTask = channel.unary_unary(
            "/agent.AgentService/SubmitTask",
            request_serializer=_struct_pb2.Struct.SerializeToString,
            response_deserializer=_struct_pb2.Struct.FromString,
        )
        self.GetTaskStatus = channel.unary_unary(
            "/agent.AgentService/GetTaskStatus",
            request_serializer=_struct_pb2.Struct.SerializeToString,
            response_deserializer=_struct_pb2.Struct.FromString,
        )
        self.ResumeTask = channel.unary_unary(
            "/agent.AgentService/ResumeTask",
            request_serializer=_struct_pb2.Struct.SerializeToString,
            response_deserializer=_struct_pb2.Struct.FromString,
        )


def add_AgentServiceServicer_to_server(servicer, server):
    rpc_handlers = {
        "Health": _grpc.unary_unary_rpc_method_handler(
            servicer.Health,
            request_deserializer=_struct_pb2.Struct.FromString,
            response_serializer=_struct_pb2.Struct.SerializeToString,
        ),
        "SubmitTask": _grpc.unary_unary_rpc_method_handler(
            servicer.SubmitTask,
            request_deserializer=_struct_pb2.Struct.FromString,
            response_serializer=_struct_pb2.Struct.SerializeToString,
        ),
        "GetTaskStatus": _grpc.unary_unary_rpc_method_handler(
            servicer.GetTaskStatus,
            request_deserializer=_struct_pb2.Struct.FromString,
            response_serializer=_struct_pb2.Struct.SerializeToString,
        ),
        "ResumeTask": _grpc.unary_unary_rpc_method_handler(
            servicer.ResumeTask,
            request_deserializer=_struct_pb2.Struct.FromString,
            response_serializer=_struct_pb2.Struct.SerializeToString,
        ),
    }
    generic_handler = _grpc.method_handlers_generic_handler(
        "agent.AgentService", rpc_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))
