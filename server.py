import grpc
import json
import sys
import time
from concurrent import futures
sys.path.insert(0, "./gen/python")

import plugin_pb2
import plugin_pb2_grpc
from plugin_pb2 import (
    PluginStatus,
    PluginType,
    DataType,
    PluginError,
    InitializeRequest,
    InitializeResponse,
    GetMetadataRequest,
    GetMetadataResponse,
    HealthRequest,
    HealthResponse,
    HandleEventRequest,
    HandleEventResponse,
    StreamRequest,
    StreamResponse,
    ShutdownRequest,
    ShutdownResponse,
)
from hand_gesture import gesture_stream, stop_stream
from grpc_reflection.v1alpha import reflection
from typing import Generator

PORT: int = 50051


class VisionPlugin(plugin_pb2_grpc.PluginServiceServicer):

    def __init__(self) -> None:
        self.plugin_id: str | None = None
        self.status: int           = PluginStatus.Value("PLUGIN_STATUS_READY")

    def Initialize(
        self,
        request: InitializeRequest,
        context: grpc.ServicerContext
    ) -> InitializeResponse:
        self.plugin_id = request.plugin_id
        print(f"[vision] initialized with id: {self.plugin_id}")
        return InitializeResponse(success=True)

    def GetMetadata(
        self,
        request: GetMetadataRequest,
        context: grpc.ServicerContext
    ) -> GetMetadataResponse:
        return GetMetadataResponse(
            plugin_id   = "vision-plugin",
            name        = "Vision Plugin",
            version     = "0.1.0",
            description = "Hand gesture detection using MediaPipe",
            type        = PluginType.Value("PLUGIN_TYPE_VISION"),
            inputs      = [],
            outputs     = [DataType.Value("DATA_TYPE_JSON")]
        )

    def Health(
        self,
        request: HealthRequest,
        context: grpc.ServicerContext
    ) -> HealthResponse:
        return HealthResponse(
            status  = self.status,
            message = "ok"
        )

    def Stream(
        self,
        request: StreamRequest,
        context: grpc.ServicerContext
    ) -> Generator[StreamResponse, None, None]:
        print("[vision] stream started")
        self.status = PluginStatus.Value("PLUGIN_STATUS_BUSY")

        try:
            for event in gesture_stream():
                if not context.is_active():
                    print("[vision] client disconnected")
                    break

                payload: bytes = json.dumps(event).encode("utf-8")

                yield StreamResponse(
                    data_type = DataType.Value("DATA_TYPE_JSON"),
                    payload   = payload
                )

        except Exception as e:
            yield StreamResponse(
                error = PluginError(
                    code    = "STREAM_ERROR",
                    message = str(e)
                )
            )

        finally:
            self.status = PluginStatus.Value("PLUGIN_STATUS_READY")
            print("[vision] stream ended")

    def HandleEvent(
        self,
        request: HandleEventRequest,
        context: grpc.ServicerContext
    ) -> HandleEventResponse:
        event_type: str = request.event_type

        if event_type == "pause":
            stop_stream()
            return HandleEventResponse(success=True)

        if event_type == "resume":
            return HandleEventResponse(success=True)

        return HandleEventResponse(
            success = False,
            error   = PluginError(
                code    = "UNKNOWN_EVENT",
                message = f"unknown event type: {event_type}"
            )
        )

    def Shutdown(
        self,
        request: ShutdownRequest,
        context: grpc.ServicerContext
    ) -> ShutdownResponse:
        print("[vision] shutting down")
        stop_stream()
        self.status = PluginStatus.Value("PLUGIN_STATUS_LOADING")
        return ShutdownResponse(success=True)


def serve() -> None:
    server: grpc.Server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4)
    )

    plugin_pb2_grpc.add_PluginServiceServicer_to_server(
        VisionPlugin(), server
    )

    SERVICE_NAMES = (
        plugin_pb2.DESCRIPTOR.services_by_name["PluginService"].full_name,
        reflection.SERVICE_NAME,
    )
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    server.add_insecure_port(f"[::]:{PORT}")
    server.start()
    print(f"[vision] server running on port {PORT}")

    try:
        while True:
            time.sleep(86400)
    except KeyboardInterrupt:
        stop_stream()
        server.stop(0)
        print("[vision] stopped")


if __name__ == "__main__":
    serve()
