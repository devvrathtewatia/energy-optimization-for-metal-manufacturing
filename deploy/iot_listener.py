"""Listens for machine readings on an MQTT topic (AWS IoT Core, or a local broker for testing) and hands each one
to a callback. Used by deploy/api.py. Never crashes the API: if the connection fails it just keeps retrying."""
import json
import logging
import ssl

import paho.mqtt.client as mqtt

log = logging.getLogger("iot")
TOPIC = "plant/cnc1/telemetry"


class Listener:
    def __init__(self, on_reading, endpoint=None, cert=None, key=None, ca=None, local_host=None,
                 topic=TOPIC, client_id="energy-api-listener"):
        self.on_reading, self.endpoint, self.cert, self.key, self.ca = on_reading, endpoint, cert, key, ca
        self.local_host, self.topic, self.client_id = local_host, topic, client_id
        self.connected, self.received, self.last_error = False, 0, None

    def start(self):
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=self.client_id)
        if self.local_host:                       # plain local broker, for testing only
            host, port = self.local_host, 1883
        else:                                     # AWS IoT Core: mutual TLS with the device certificate
            c.tls_set(ca_certs=self.ca or None, certfile=self.cert, keyfile=self.key, tls_version=ssl.PROTOCOL_TLS_CLIENT)
            host, port = self.endpoint, 8883

        def on_connect(client, userdata, flags, reason_code, properties=None):
            self.connected = not reason_code.is_failure
            if self.connected:
                client.subscribe(self.topic, qos=1)      # re-subscribe after every (re)connect
                self.last_error = None
                log.info("connected, subscribed to %s", self.topic)
            else:
                self.last_error = f"connect refused: {reason_code}"
                log.error(self.last_error)

        def on_disconnect(client, userdata, disconnect_flags, reason_code, properties=None):
            self.connected = False
            log.warning("disconnected: %s", reason_code)

        def on_message(client, userdata, msg):
            try:
                self.on_reading(json.loads(msg.payload))
                self.received += 1
            except Exception as e:  # noqa: BLE001
                self.last_error = f"bad message: {e}"
                log.warning(self.last_error)

        c.on_connect, c.on_disconnect, c.on_message = on_connect, on_disconnect, on_message
        c.reconnect_delay_set(1, 30)
        c.connect_async(host, port, keepalive=30)
        c.loop_start()
        self.client = c
        return self
