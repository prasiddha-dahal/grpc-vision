import grpc
import json
import sys
import time
from concurrent import futures

sys.path.insert(0, "./gen/python")

import plugin_pb2
import plugin_pb2_grpc

from hand_gesture import gesture_stream, stop_stream

PORT = 50051


class VisionPlugin(plugin_pb2_grpc.PluginServiceServicer):

    def __init__(self):
        self.plugin_id = None
        self.status    = plugin_pb2.PLUGIN_STATUS_READY

    def Initialize(self, request, context):
        self.plugin_id = request.plugin_id
        print(f"[vision] initialized with id: {self.plugin_id}")
        return plugin_pb2.InitializeResponse(success=True)

    def GetMetadata(self, request, context):
        return plugin_pb2.GetMetadataResponse(
            plugin_id   = "vision-plugin",
            name        = "Vision Plugin",
            version     = "0.1.0",
            description = "Hand gesture detection using MediaPipe",
            type        = plugin_pb2.PLUGIN_TYPE_VISION,
            inputs      = [],
            outputs     = [plugin_pb2.DATA_TYPE_JSON]
        )

    def Health(self, request, context):
        return plugin_pb2.HealthResponse(
            status  = self.status,
            message = "ok"
        )

    def Stream(self, request, context):
        print("[vision] stream started")
        self.status = plugin_pb2.PLUGIN_STATUS_BUSY

        try:
            for event in gesture_stream():
                if not context.is_active():
                    print("[vision] client disconnected")
                    break

                payload = json.dumps(event).encode("utf-8")

                yield plugin_pb2.StreamResponse(
                    data_type = plugin_pb2.DATA_TYPE_JSON,
                    payload   = payload
                )

        except Exception as e:
            yield plugin_pb2.StreamResponse(
                error = plugin_pb2.PluginError(
                    code    = "STREAM_ERROR",
                    message = str(e)
                )
            )

        finally:
            self.status = plugin_pb2.PLUGIN_STATUS_READY
            print("[vision] stream ended")

    def HandleEvent(self, request, context):
        if request.event_type == "pause":
            stop_stream()
            return plugin_pb2.HandleEventResponse(success=True)

        if request.event_type == "resume":
            return plugin_pb2.HandleEventResponse(success=True)

        return plugin_pb2.HandleEventResponse(
            success = False,
            error   = plugin_pb2.PluginError(
                code    = "UNKNOWN_EVENT",
                message = f"unknown event type: {request.event_type}"
            )
        )

    def Shutdown(self, request, context):
        print("[vision] shutting down")
        stop_stream()
        self.status = plugin_pb2.PLUGIN_STATUS_LOADING
        return plugin_pb2.ShutdownResponse(success=True)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))

    plugin_pb2_grpc.add_PluginServiceServicer_to_server(
        VisionPlugin(), server
    )

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
