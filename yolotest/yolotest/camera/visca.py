import socket
from ..viscaMappings12 import FOVLOOKUPP12, P12STEP

VISCA_PORT = 5500

# function to change the FOV of a camera
def change_fov(connection: socket.socket, zoom: str) -> None:
    # send visca command
    connection.sendall(bytes.fromhex(f"81 01 04 47 {zoom} FF"))
    responses = bytearray()
    # poll until we recieve a completion response
    while True:
        data = connection.recv(1024)
        if not data:
            raise ConnectionError("Connection closed by camera")
        responses.extend(data)
        # investigate all active responses
        while 0xFF in responses:
            end = responses.index(0xFF)
            response, responses = responses[: end + 1], responses[end + 1 :]
            # exit when we receive completion
            if response.startswith(b"\x90\x51"):
                return

# function to obtain current FOV value
def get_current_fov(connection: socket.socket) -> str:
    # send visca command to get current zoom
    connection.sendall(bytes.fromhex("81 09 04 47 FF"))
    responses = bytearray()
    while True:
        # append new data to responses
        responses.extend(connection.recv(1024))
        # peel off the last complete response
        while 0xFF in responses:
            end = responses.index(0xFF)
            msg, responses = bytes(responses[:end + 1]), responses[end + 1 :]
            # Ignore ACK
            if msg.startswith(b"\x90\x41"):
                continue
            # Zoom inquiry response
            if msg.startswith(b"\x90\x50"):
                # extract zoom value from response
                visca = int("".join(f"{b:x}"[-1] for b in msg[2:6]), 16)
                print(f"Current VISCA zoom value: {visca}")
                index = visca // P12STEP
                return FOVLOOKUPP12[index]
