#!/usr/bin/env python3
"""
Supervisord event listener for fail-fast behavior.

If OVMS or nginx exits unexpectedly (not orchestrator, which is expected
to exit 0 after setup), this listener tells supervisord to shut down,
causing the container to exit so the orchestrator can reschedule it.
"""

import sys
import os


def write_stdout(msg):
    sys.stdout.write(msg)
    sys.stdout.flush()


def write_stderr(msg):
    sys.stderr.write(msg)
    sys.stderr.flush()


def main():
    while True:
        # Supervisord sends READY\n, we respond with READY\n
        write_stdout("READY\n")

        # Read event header
        line = sys.stdin.readline()
        if not line:
            break

        # Parse header
        headers = dict(pair.split(":") for pair in line.strip().split() if ":" in pair)

        # Read event payload
        payload_len = int(headers.get("len", 0))
        payload = sys.stdin.read(payload_len) if payload_len > 0 else ""

        # Parse payload
        payload_data = dict(
            pair.split(":")
            for pair in payload.strip().split()
            if ":" in pair
        )

        process_name = payload_data.get("processname", "unknown")
        event_type = headers.get("eventname", "UNKNOWN")

        # The orchestrator is expected to exit 0 — ignore it
        if process_name == "orchestrator":
            expected_exit = payload_data.get("expected", "0")
            if expected_exit == "1":
                write_stderr(
                    f"INFO: Orchestrator exited as expected (exit code 0)\n"
                )
                write_stdout("RESULT 2\nOK")
                continue

        # For OVMS and nginx, any unexpected exit is fatal
        if process_name in ("ovms", "nginx"):
            exit_code = payload_data.get("exitcode", "?")
            write_stderr(
                f"FATAL: Process '{process_name}' exited unexpectedly "
                f"(event={event_type}, exit_code={exit_code}). "
                f"Shutting down container.\n"
            )

            # Tell supervisord to stop all processes
            import xmlrpc.client
            try:
                server = xmlrpc.client.ServerProxy(
                    "http://localhost",
                    transport=xmlrpc.client.Transport(),
                )
                # Use Unix socket
                import http.client
                import socket

                class UnixStreamHTTPConnection(http.client.HTTPConnection):
                    def connect(self):
                        self.sock = socket.socket(
                            socket.AF_UNIX, socket.SOCK_STREAM
                        )
                        self.sock.connect("/var/run/supervisor.sock")

                class UnixStreamTransport(xmlrpc.client.Transport):
                    def make_connection(self, host):
                        return UnixStreamHTTPConnection(host)

                server = xmlrpc.client.ServerProxy(
                    "http://localhost",
                    transport=UnixStreamTransport(),
                )
                server.supervisor.shutdown()
            except Exception as e:
                write_stderr(f"Failed to shutdown supervisor via XML-RPC: {e}\n")
                # Fallback: exit this listener, which supervisord should notice
                os._exit(1)

        write_stdout("RESULT 2\nOK")


if __name__ == "__main__":
    main()
