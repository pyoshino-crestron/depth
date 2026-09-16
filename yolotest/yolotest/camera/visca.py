import socket

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
